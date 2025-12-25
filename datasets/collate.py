"""
Collate function for MAE-style masked training
"""

import torch
import numpy as np
from typing import Dict, List


class MultiScaleMaskedCollate:
    """
    Collate function for MAE-style masked pretraining.

    Features:
    - Fixed sequence length
    - Patch-level masking for images (like ViT MAE)
    - Temporal masking for vectors
    - Support for unified/independent mask strategies across modalities
    """

    def __init__(
        self,
        # Sequence length (fixed)
        seq_len: int = 90,
        # Mask parameters
        mask_ratio: float = 0.75,  # Ratio to mask (0.75 like MAE paper)
        # Image patch parameters
        patch_size: int = 10,  # Each patch is 10x10 pixels
        image_height: int = 290,
        image_width: int = 180,
        # Land mask
        land_mask_path: str = None,  # Path to land mask file
        land_threshold: float = 0.5,  # Minimum land ratio for valid patch
        # Modality mask strategy
        mask_mode: str = 'unified',  # 'independent' or 'unified'
        # Mode
        mode: str = 'train',  # 'train', 'val', 'test'
    ):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.land_threshold = land_threshold
        self.mask_mode = mask_mode
        self.mode = mode

        # Calculate number of patches
        self.num_patches_h = image_height // patch_size  # 290 // 10 = 29
        self.num_patches_w = image_width // patch_size   # 180 // 10 = 18
        self.num_patches = self.num_patches_h * self.num_patches_w  # 29 * 18 = 522

        # Load and process land mask
        self.valid_patch_indices = self._process_land_mask(land_mask_path)

        # Modality lists
        self.image_modalities = ['precip', 'soil', 'temp']
        self.vector_modalities = ['evap', 'riverflow']
        self.all_modalities = self.image_modalities + self.vector_modalities

    def _process_land_mask(self, land_mask_path: str) -> np.ndarray:
        """
        Process land mask to identify valid patches

        Args:
            land_mask_path: Path to land mask file (290x180)

        Returns:
            valid_patch_indices: Array of valid patch indices
        """
        if land_mask_path is None:
            # If no land mask provided, all patches are valid
            return np.arange(self.num_patches)

        import torch
        # Load land mask
        land_mask = torch.load(land_mask_path).numpy()  # [290, 180]

        # Calculate land coverage for each patch
        valid_patches = []
        patch_idx = 0

        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Extract patch region
                patch = land_mask[
                    i*self.patch_size:(i+1)*self.patch_size,
                    j*self.patch_size:(j+1)*self.patch_size
                ]

                # Calculate land ratio in this patch
                land_ratio = patch.sum() / (self.patch_size * self.patch_size)

                # Check if patch meets threshold
                if land_ratio >= self.land_threshold:
                    valid_patches.append(patch_idx)

                patch_idx += 1

        valid_patches = np.array(valid_patches, dtype=np.int64)
        print(f"Land mask loaded: {len(valid_patches)}/{self.num_patches} patches are valid "
              f"(>={self.land_threshold*100:.0f}% land)")

        return valid_patches

    def __call__(self, batch_list: List[Dict]) -> Dict:
        """
        Process batch

        Args:
            batch_list: List of samples from Dataset

        Returns:
            batch_dict: {
                # Data
                'precip': [B, T, 290, 180],
                'soil': [B, T, 290, 180],
                'temp': [B, T, 290, 180],
                'evap': [B, T],
                'riverflow': [B, T],
                'static_attr': [B, num_features],

                # Masks (True = positions to predict)
                # For images: [B, T, num_patches] where num_patches = 522
                'precip_mask': [B, T, 522],
                'soil_mask': [B, T, 522],
                'temp_mask': [B, T, 522],
                # For vectors: [B, T] (temporal masking)
                'evap_mask': [B, T],
                'riverflow_mask': [B, T],

                # Metadata
                'catchment_ids': [B],
                'seq_len': int,
            }
        """
        B = len(batch_list)
        seq_len = self.seq_len

        # Step 1: Truncate data to fixed sequence length
        truncated_batch = []
        for sample in batch_list:
            truncated = {}
            for key, val in sample.items():
                if key in self.all_modalities:
                    truncated[key] = val[:seq_len]
                else:
                    truncated[key] = val
            truncated_batch.append(truncated)

        # Step 2: Generate masks
        if self.mode == 'train':
            masks = self._generate_masks(B, seq_len)
        else:
            # No masking for validation/test
            masks = {
                # Image masks: [B, T, num_patches]
                **{mod: np.zeros((B, seq_len, self.num_patches), dtype=bool)
                   for mod in self.image_modalities},
                # Vector masks: [B, T]
                **{mod: np.zeros((B, seq_len), dtype=bool)
                   for mod in self.vector_modalities}
            }

        # Step 3: Stack into batch
        batch_dict = {}

        # Image modalities
        for mod in self.image_modalities:
            batch_dict[mod] = torch.stack([
                torch.from_numpy(s[mod]) for s in truncated_batch
            ]).float()  # [B, T, 290, 180]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])  # [B, T, num_patches]

        # Vector modalities
        for mod in self.vector_modalities:
            batch_dict[mod] = torch.stack([
                torch.from_numpy(s[mod]) for s in truncated_batch
            ]).float()  # [B, T]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])  # [B, T]

        # Static attributes
        batch_dict['static_attr'] = torch.stack([
            s['static_attr'] for s in truncated_batch
        ])  # [B, num_features]

        # Metadata
        batch_dict['catchment_ids'] = torch.tensor([
            s['catchment_id'] for s in truncated_batch
        ], dtype=torch.long)
        batch_dict['seq_len'] = seq_len

        return batch_dict

    def _generate_masks(self, B: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        Generate masks for each modality

        Args:
            B: batch size
            seq_len: sequence length

        Returns:
            Dictionary with:
            - Image modalities: mask [B, T, num_patches]
            - Vector modalities: mask [B, T]
        """
        masks = {}

        if self.mask_mode == 'unified':
            # All modalities use the same mask pattern
            # For images: generate patch-level mask [B, T, num_patches]
            image_mask = self._generate_image_mask(B, seq_len)
            for mod in self.image_modalities:
                masks[mod] = image_mask.copy()

            # For vectors: generate temporal mask [B, T]
            vector_mask = self._generate_vector_mask(B, seq_len)
            for mod in self.vector_modalities:
                masks[mod] = vector_mask.copy()

        elif self.mask_mode == 'independent':
            # Each modality has independent mask
            for mod in self.image_modalities:
                masks[mod] = self._generate_image_mask(B, seq_len)
            for mod in self.vector_modalities:
                masks[mod] = self._generate_vector_mask(B, seq_len)

        return masks

    def _generate_image_mask(self, B: int, seq_len: int) -> np.ndarray:
        """
        Generate patch-level mask for images (MAE-style)
        Only masks valid patches (land patches)

        Args:
            B: batch size
            seq_len: sequence length

        Returns:
            mask: [B, T, num_patches], True = positions to predict
        """
        masks = []

        for _ in range(B):
            # For each sample, generate mask for each timestep
            sample_masks = []
            for _ in range(seq_len):
                # Initialize mask: False for all patches
                patch_mask = np.zeros(self.num_patches, dtype=bool)

                # Only mask from valid patches
                num_valid = len(self.valid_patch_indices)
                if num_valid > 0:
                    # Calculate how many valid patches to mask
                    num_to_mask = int(num_valid * self.mask_ratio)
                    num_to_mask = max(1, num_to_mask)  # At least mask 1 patch

                    # Randomly select which valid patches to mask
                    masked_valid_indices = np.random.choice(
                        num_valid,
                        size=num_to_mask,
                        replace=False
                    )

                    # Convert to actual patch indices
                    masked_patch_indices = self.valid_patch_indices[masked_valid_indices]

                    # Set mask
                    patch_mask[masked_patch_indices] = True

                sample_masks.append(patch_mask)

            masks.append(np.stack(sample_masks, axis=0))  # [T, num_patches]

        return np.stack(masks, axis=0)  # [B, T, num_patches]

    def _generate_vector_mask(self, B: int, seq_len: int) -> np.ndarray:
        """
        Generate temporal mask for vectors

        Randomly selects mask_ratio of timesteps to mask.

        Args:
            B: batch size
            seq_len: sequence length

        Returns:
            mask: [B, T], True = positions to predict
        """
        masks = []

        for _ in range(B):
            # Random temporal mask: select mask_ratio of days to mask
            num_to_mask = int(seq_len * self.mask_ratio)
            time_mask = np.zeros(seq_len, dtype=bool)
            masked_indices = np.random.choice(
                seq_len,
                size=num_to_mask,
                replace=False
            )
            time_mask[masked_indices] = True
            masks.append(time_mask)

        return np.stack(masks, axis=0)  # [B, T]
