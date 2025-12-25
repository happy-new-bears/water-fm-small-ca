"""
Multi-modal MAE Main Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
import os

from .image_encoder import ImageModalityEncoder
from .vector_encoder import VectorModalityEncoder
from .image_decoder import ImageModalityDecoder
from .vector_decoder import VectorModalityDecoder
from .layers import patchify, unpatchify


class MultiModalMAE(nn.Module):
    """
    Multi-modal Masked Autoencoder for Hydrology Data

    Supports 5 modalities:
        - 3 Image modalities: precipitation, soil_moisture, temperature
        - 2 Vector modalities: evaporation, riverflow

    Each modality has independent encoder and decoder.
    Vector encoders use FiLM mechanism to fuse static attributes.

    Args:
        config: MAEConfig object with model configuration
        valid_patch_indices: Tensor of valid land patch indices (94 patches)
    """

    def __init__(self, config, valid_patch_indices: Tensor = None):
        super().__init__()

        self.config = config

        # Store valid patch indices for land mask
        if valid_patch_indices is not None:
            self.register_buffer('valid_patch_indices', valid_patch_indices)
        else:
            # Default: use all patches
            num_patches = (config.image_height // config.patch_size) * \
                         (config.image_width // config.patch_size)
            self.register_buffer(
                'valid_patch_indices',
                torch.arange(num_patches, dtype=torch.long)
            )

        # ========== Image Encoders ==========
        self.precip_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
        )

        self.soil_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
        )

        self.temp_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
        )

        # ========== Vector Encoders (with FiLM) ==========
        # Use max_len = 120 for vector encoder positional encoding
        # This is sufficient for typical visible sequence lengths
        vector_max_len = 120

        self.evap_encoder = VectorModalityEncoder(
            in_feat=1,
            stat_dim=config.static_attr_dim,
            d_model=config.d_model,
            n_layers=config.vec_encoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            max_len=vector_max_len,  # Use 120 instead of calculated 2280
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            use_fm_layers=config.use_fm_layers,      # Phase 2
            use_input=config.use_input,              # Phase 2
            patch_size=config.vector_patch_size,     # NEW: Vector patch size
        )

        self.riverflow_encoder = VectorModalityEncoder(
            in_feat=1,
            stat_dim=config.static_attr_dim,
            d_model=config.d_model,
            n_layers=config.vec_encoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            max_len=vector_max_len,  # Use 120 instead of calculated 2280
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            use_fm_layers=config.use_fm_layers,      # Phase 2
            use_input=config.use_input,              # Phase 2
            patch_size=config.vector_patch_size,     # NEW: Vector patch size
        )

        # ========== Image Decoders ==========
        num_patches = (config.image_height // config.patch_size) * \
                     (config.image_width // config.patch_size)

        self.precip_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        self.soil_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        self.temp_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        # ========== Vector Decoders ==========
        self.evap_decoder = VectorModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            max_time_steps=config.max_time_steps,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            num_encoder_layers=config.vec_encoder_layers,  # Phase 2
        )

        self.riverflow_decoder = VectorModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            max_time_steps=config.max_time_steps,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            num_encoder_layers=config.vec_encoder_layers,  # Phase 2
        )

    def forward(self, batch: Dict) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass

        Args:
            batch: Dictionary with:
                - 'precip', 'soil', 'temp': [B, T, H, W] image data
                - 'evap', 'riverflow': [B, T] vector data
                - 'static_attr': [B, stat_dim] static attributes
                - '*_mask': [B, T, num_patches] or [B, T] masks (True=masked)

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses for each modality
        """

        # ===== Encode all modalities =====
        # Image modalities
        precip_token, precip_mask_info = self.precip_encoder(
            batch['precip'], batch['precip_mask']
        )
        soil_token, soil_mask_info = self.soil_encoder(
            batch['soil'], batch['soil_mask']
        )
        temp_token, temp_mask_info = self.temp_encoder(
            batch['temp'], batch['temp_mask']
        )

        # Vector modalities (with FiLM)
        evap_token, evap_mask_info = self.evap_encoder(
            batch['evap'], batch['static_attr'], batch['evap_mask']
        )
        riverflow_token, riverflow_mask_info = self.riverflow_encoder(
            batch['riverflow'], batch['static_attr'], batch['riverflow_mask']
        )

        # ===== Decode all modalities =====
        # Image modalities
        precip_pred = self.precip_decoder(precip_token, precip_mask_info)
        soil_pred = self.soil_decoder(soil_token, soil_mask_info)
        temp_pred = self.temp_decoder(temp_token, temp_mask_info)

        # Vector modalities
        evap_pred = self.evap_decoder(evap_token, evap_mask_info)
        riverflow_pred = self.riverflow_decoder(riverflow_token, riverflow_mask_info)

        # ===== Compute losses =====
        loss_dict = {}

        # Image losses
        loss_dict['precip_loss'] = self._compute_image_loss(
            precip_pred, batch['precip'], batch['precip_mask']
        )
        loss_dict['soil_loss'] = self._compute_image_loss(
            soil_pred, batch['soil'], batch['soil_mask']
        )
        loss_dict['temp_loss'] = self._compute_image_loss(
            temp_pred, batch['temp'], batch['temp_mask']
        )

        # Vector losses
        loss_dict['evap_loss'] = self._compute_vector_loss(
            evap_pred, batch['evap'], batch['evap_mask']
        )
        loss_dict['riverflow_loss'] = self._compute_vector_loss(
            riverflow_pred, batch['riverflow'], batch['riverflow_mask']
        )

        # Total loss (simple sum)
        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def _compute_image_loss(
        self,
        pred_patches: Tensor,
        target_img: Tensor,
        mask: Tensor
    ) -> Tensor:
        """
        Compute reconstruction loss for image modality

        Args:
            pred_patches: [B, T, num_patches, patch_dim] predicted patches
            target_img: [B, T, H, W] target image
            mask: [B, T, num_patches] bool mask (True=masked)

        Returns:
            Scalar loss (only on masked AND valid land patches)
        """
        # Patchify target
        target_patches = patchify(
            target_img,
            patch_size=self.config.patch_size
        )  # [B, T, num_patches, patch_dim]

        # MSE loss
        loss = F.mse_loss(
            pred_patches, target_patches, reduction='none'
        )  # [B, T, num_patches, patch_dim]

        # Average over patch dimension
        loss = loss.mean(dim=-1)  # [B, T, num_patches]

        # Create valid patch mask [1, 1, num_patches]
        # Only 94 patches are valid land patches
        valid_mask = torch.zeros(1, 1, pred_patches.shape[2], device=loss.device)
        valid_mask[0, 0, self.valid_patch_indices] = 1.0

        # Combine with temporal mask: only compute loss on masked AND valid patches
        # mask: [B, T, num_patches] - True for masked positions
        # valid_mask: [1, 1, num_patches] - 1.0 for valid land patches
        combined_mask = mask.float() * valid_mask  # [B, T, num_patches]

        # Only compute loss on masked AND valid patches
        masked_loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        return masked_loss

    def _compute_vector_loss(
        self,
        pred_vec: Tensor,
        target_vec: Tensor,
        mask: Tensor
    ) -> Tensor:
        """
        Compute reconstruction loss for vector modality

        Args:
            pred_vec: [B, num_catchments, T] predicted values (unpatchified)
            target_vec: [B, num_patches, patch_size, T] target values (patchified)
            mask: [B, num_patches, T] bool mask (True=masked patch)

        Returns:
            Scalar loss (only on masked positions)
        """
        # Unpatchify target: [B, num_patches, patch_size, T] -> [B, num_catchments, T]
        B, num_patches, patch_size, T = target_vec.shape
        num_padded = num_patches * patch_size

        # Flatten patch dimension: [B, num_patches, patch_size, T] -> [B, num_padded, T]
        target_flat = target_vec.reshape(B, num_padded, T)

        # Remove padding to match actual number of catchments
        num_actual = pred_vec.shape[1]  # Should be 604
        target_unpatch = target_flat[:, :num_actual, :]  # [B, 604, T]

        # Unpatchify mask: [B, num_patches, T] -> [B, num_catchments, T]
        # Expand mask to cover all catchments in each patch
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, patch_size, -1)  # [B, num_patches, patch_size, T]
        mask_flat = mask_expanded.reshape(B, num_padded, T)  # [B, num_padded, T]
        mask_unpatch = mask_flat[:, :num_actual, :]  # [B, 604, T]

        # MSE loss
        loss = F.mse_loss(
            pred_vec, target_unpatch, reduction='none'
        )  # [B, num_catchments, T]

        # Only compute loss on masked positions
        masked_loss = (loss * mask_unpatch.float()).sum() / (mask_unpatch.sum() + 1e-8)

        return masked_loss


if __name__ == '__main__':
    """Unit test for MultiModalMAE"""

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.image_encoder import ImageModalityEncoder
    from models.vector_encoder import VectorModalityEncoder
    from models.image_decoder import ImageModalityDecoder
    from models.vector_decoder import VectorModalityDecoder
    from models.layers import patchify
    from configs.mae_config import MAEConfig

    print("=" * 60)
    print("Testing MultiModalMAE")
    print("=" * 60)

    # Create config
    config = MAEConfig()

    # Simulate valid patch indices
    num_valid = 94
    valid_patch_indices = torch.randperm(522)[:num_valid].sort()[0]

    # Create model
    model = MultiModalMAE(config, valid_patch_indices)

    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test batch (smaller size to avoid OOM)
    B, T, H, W = 2, 10, 290, 180
    batch = {
        # Image modalities
        'precip': torch.randn(B, T, H, W),
        'soil': torch.randn(B, T, H, W),
        'temp': torch.randn(B, T, H, W),

        # Vector modalities
        'evap': torch.randn(B, T),
        'riverflow': torch.randn(B, T),

        # Static attributes
        'static_attr': torch.randn(B, 11),

        # Masks (75% masked)
        'precip_mask': torch.rand(B, T, 522) < 0.75,
        'soil_mask': torch.rand(B, T, 522) < 0.75,
        'temp_mask': torch.rand(B, T, 522) < 0.75,
        'evap_mask': torch.rand(B, T) < 0.75,
        'riverflow_mask': torch.rand(B, T) < 0.75,
    }

    print(f"\n✓ Test batch created")
    print(f"  Image shape: {batch['precip'].shape}")
    print(f"  Vector shape: {batch['evap'].shape}")

    # Forward pass
    total_loss, loss_dict = model(batch)

    print(f"\n✓ Forward pass successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"    {key}: {value.item():.4f}")

    # Backward pass
    total_loss.backward()

    print(f"\n✓ Backward pass successful")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
