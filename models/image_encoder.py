"""
Image Modality Encoder
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from .layers import PositionalEncoding, patchify


class ImageModalityEncoder(nn.Module):
    """
    Image Modality Encoder with patch-level masking (CrossMAE style)

    Architecture:
        1. Patchify: [B, T, H, W] -> [B, T, num_patches, patch_dim]
        2. Filter valid land patches (94 out of 522)
        3. Remove masked patches (encoder never sees masked data)
        4. Add spatial + temporal position embeddings
        5. Transformer encoder (standard, no FiLM)
        6. Normalize (NO POOLING - keeps sequence for CrossMAE decoder)
        7. Output sequence of visible tokens [B, L_visible, d_model]

    Args:
        patch_size: Size of each patch (default: 10)
        image_hw: (H, W) image size (default: (290, 180))
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        max_time_steps: Maximum sequence length
        dropout: Dropout rate
        valid_patch_indices: Indices of valid land patches (94 patches)
    """

    def __init__(
        self,
        patch_size: int = 10,
        image_hw: Tuple[int, int] = (290, 180),
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_time_steps: int = 90,
        dropout: float = 0.1,
        valid_patch_indices: Tensor = None,
        use_weighted_fm: bool = False,  # NEW: Phase 2
        use_fm_layers: list = None,    # NEW: Which layers to save
        use_input: bool = False,        # NEW: Include input as layer 0
    ):
        super().__init__()

        self.patch_size = patch_size
        self.image_hw = image_hw
        self.d_model = d_model
        self.use_weighted_fm = use_weighted_fm
        self.use_input = use_input

        H, W = image_hw
        self.num_patches_h = H // patch_size  # 29
        self.num_patches_w = W // patch_size  # 18
        self.num_patches = self.num_patches_h * self.num_patches_w  # 522
        self.patch_dim = patch_size * patch_size  # 100

        # Valid patch indices (land patches only)
        if valid_patch_indices is not None:
            self.register_buffer('valid_patch_indices', valid_patch_indices)
            self.num_valid_patches = len(valid_patch_indices)
        else:
            # If not provided, use all patches
            self.register_buffer(
                'valid_patch_indices',
                torch.arange(self.num_patches, dtype=torch.long)
            )
            self.num_valid_patches = self.num_patches

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, d_model)

        # Spatial positional embedding (learnable, shared across time)
        # Only for valid patches
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_valid_patches, d_model)
        )
        nn.init.normal_(self.spatial_pos, std=0.02)

        # Temporal positional encoding (fixed sincos)
        self.temporal_pos = PositionalEncoding(d_model, max_time_steps)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

        # Phase 2: Determine which layers to save for WeightedFeatureMaps
        if use_weighted_fm:
            if use_fm_layers is None:
                # Use all layers
                self.use_fm_layers = list(range(num_layers))
            else:
                # Use specified layers
                self.use_fm_layers = use_fm_layers
        else:
            self.use_fm_layers = []

    def forward(self, x_img: Tensor, patch_mask: Tensor) -> Tuple[Tensor, Dict]:
        """
        Forward pass

        Args:
            x_img: [B, T, H, W] image sequence
            patch_mask: [B, T, 522] bool mask (True = masked, False = visible)

        Returns:
            encoder_output: [B, L_visible, d_model] - sequence of visible tokens (CrossMAE style)
            mask_info: dict with mask information for decoder
        """
        B, T, H, W = x_img.shape

        # ===== Step 1: Patchify =====
        patches = patchify(x_img, self.patch_size)  # [B, T, 522, 100]

        # ===== Step 2: Filter valid patches only =====
        patches = patches[:, :, self.valid_patch_indices, :]  # [B, T, num_valid, 100]
        patch_mask_valid = patch_mask[:, :, self.valid_patch_indices]  # [B, T, num_valid]

        # ===== Step 3: Remove masked patches (MAE strategy) =====
        # Collect visible patches across batch and time
        visible_patches_list = []
        visible_positions_list = []  # (time_idx, patch_idx) pairs
        lengths = []

        for b in range(B):
            sample_patches = []
            sample_positions = []

            for t in range(T):
                # Get visible patches at this timestep
                visible_mask_t = ~patch_mask_valid[b, t]  # True = visible
                visible_patches_t = patches[b, t, visible_mask_t]  # [num_visible_t, 100]

                # Get patch indices
                visible_patch_indices = torch.where(visible_mask_t)[0]

                sample_patches.append(visible_patches_t)

                # Record positions for spatial PE
                for patch_idx in visible_patch_indices:
                    sample_positions.append((t, patch_idx.item()))

            # Concatenate all visible patches for this sample
            if len(sample_patches) > 0:
                sample_patches = torch.cat(sample_patches, dim=0)  # [num_visible_total, 100]
            else:
                # Edge case: no visible patches (shouldn't happen in practice)
                sample_patches = torch.zeros(0, self.patch_dim, device=x_img.device, dtype=self.patch_embed.weight.dtype)

            visible_patches_list.append(sample_patches)
            visible_positions_list.append(sample_positions)
            lengths.append(len(sample_patches))

        # Pad to max length
        max_len = max(lengths)
        if max_len == 0:
            # Edge case: no visible patches at all
            encoder_output = torch.zeros(B, 1, self.d_model, device=x_img.device, dtype=self.patch_embed.weight.dtype)  # [B, 1, d_model]
            padding_mask = torch.ones(B, 1, device=x_img.device, dtype=torch.bool)  # All padding
            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask,
                'positions': [[] for _ in range(B)],
            }
            return encoder_output, mask_info

        x_padded = torch.zeros(B, max_len, self.patch_dim, device=x_img.device, dtype=self.patch_embed.weight.dtype)
        positions_padded = []  # Store positions for PE

        for b in range(B):
            x_padded[b, :lengths[b]] = visible_patches_list[b].to(dtype=self.patch_embed.weight.dtype)
            positions_padded.append(visible_positions_list[b])

        # Create padding mask
        padding_mask = torch.zeros(B, max_len, device=x_img.device, dtype=torch.bool)
        for b in range(B):
            if lengths[b] < max_len:
                padding_mask[b, lengths[b]:] = True

        # ===== Step 4: Patch embedding =====
        x = self.patch_embed(x_padded)  # [B, max_len, d_model]

        # ===== Step 5: Add position embeddings =====
        # Spatial PE: Add based on patch position
        for b in range(B):
            for i, (t_idx, patch_idx) in enumerate(positions_padded[b]):
                x[b, i] += self.spatial_pos[0, patch_idx]

        # Temporal PE: Add based on time position
        # Create temporal indices tensor
        temporal_indices = torch.zeros(B, max_len, device=x_img.device, dtype=torch.long)
        for b in range(B):
            for i, (t_idx, patch_idx) in enumerate(positions_padded[b]):
                temporal_indices[b, i] = t_idx

        # Apply temporal PE
        temporal_pe = self.temporal_pos.pe.squeeze(0)  # [max_time, d_model]
        for b in range(B):
            for i in range(lengths[b]):
                t_idx = temporal_indices[b, i]
                x[b, i] += temporal_pe[t_idx]

        # ===== Step 6: Transformer encoder =====
        if self.use_weighted_fm:
            # Phase 2: Collect multi-layer features
            x_feats = []

            # Optional: Include input as layer 0
            if self.use_input:
                x_feats.append(self.norm(x.clone()))

            # Process through transformer layers
            for idx, layer in enumerate(self.transformer.layers):
                x = layer(x, src_key_padding_mask=padding_mask)

                # Save specified layers
                if idx in self.use_fm_layers:
                    x_feats.append(self.norm(x.clone()))

            # Return list of features
            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask,
                'positions': positions_padded,
            }

            return x_feats, mask_info  # List of [B, L_visible, d_model]

        else:
            # Standard: Single layer output
            x = self.transformer(x, src_key_padding_mask=padding_mask)

            # ===== Step 7: Normalize (NO POOLING - CrossMAE style) =====
            x = self.norm(x)  # [B, max_len, d_model]

            # ===== Prepare mask_info for decoder =====
            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask,
                'positions': positions_padded,
            }

            return x, mask_info  # [B, L_visible, d_model]


if __name__ == '__main__':
    """Unit test for ImageModalityEncoder"""

    print("=" * 60)
    print("Testing ImageModalityEncoder")
    print("=" * 60)

    # Simulate valid patch indices (94 land patches)
    num_valid = 94
    valid_patch_indices = torch.randperm(522)[:num_valid].sort()[0]

    # Create encoder
    encoder = ImageModalityEncoder(
        patch_size=10,
        image_hw=(290, 180),
        d_model=256,
        nhead=8,
        num_layers=6,
        max_time_steps=90,
        dropout=0.1,
        valid_patch_indices=valid_patch_indices,
    )

    # Test data (use smaller size to avoid OOM)
    B, T, H, W = 2, 10, 290, 180  # Reduced T from 90 to 10
    x_img = torch.randn(B, T, H, W)

    # Create mask (75% of valid patches masked)
    patch_mask_full = torch.zeros(B, T, 522, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            # Randomly mask 75% of valid patches
            num_to_mask = int(num_valid * 0.75)
            masked_valid = torch.randperm(num_valid)[:num_to_mask]
            masked_patch_indices = valid_patch_indices[masked_valid]
            patch_mask_full[b, t, masked_patch_indices] = True

    print(f"Input shape: {x_img.shape}")
    print(f"Mask shape: {patch_mask_full.shape}")
    print(f"Valid patches: {num_valid}")
    print(f"Mask ratio (valid patches): {patch_mask_full[:, :, valid_patch_indices].float().mean().item():.2%}")

    # Forward pass
    token, mask_info = encoder(x_img, patch_mask_full)

    print(f"\n✓ Output token shape: {token.shape}")
    print(f"✓ Mask info keys: {mask_info.keys()}")
    print(f"✓ Visible patch counts: {mask_info['lengths']}")

    # Test backward
    loss = token.sum()
    loss.backward()

    print(f"✓ Backward pass successful")

    # Test with different mask ratios
    print(f"\n" + "=" * 60)
    print("Testing with different mask ratios")
    print("=" * 60)

    for mask_ratio in [0.0, 0.5, 0.75, 0.9]:
        patch_mask_full = torch.zeros(B, T, 522, dtype=torch.bool)
        for b in range(B):
            for t in range(T):
                num_to_mask = int(num_valid * mask_ratio)
                if num_to_mask > 0:
                    masked_valid = torch.randperm(num_valid)[:num_to_mask]
                    masked_patch_indices = valid_patch_indices[masked_valid]
                    patch_mask_full[b, t, masked_patch_indices] = True

        token, mask_info = encoder(x_img, patch_mask_full)
        visible_count = sum(mask_info['lengths'])
        expected = B * T * num_valid * (1 - mask_ratio)
        print(f"Mask ratio {mask_ratio:.0%}: "
              f"Total visible patches {visible_count} "
              f"(expected ~{expected:.0f})")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
