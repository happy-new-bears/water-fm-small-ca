"""
Vector Modality Encoder with FiLM mechanism
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from .layers import PositionalEncoding, FiLMEncoderLayer
from .spatial_aggregation import SpatialAggregation


class VectorModalityEncoder(nn.Module):
    """
    Vector Modality Encoder with FiLM mechanism for static attribute fusion (CrossMAE style)

    Architecture:
        0. (Optional) Spatial aggregation: [B, num_catchments, T] -> [B, num_patches, T]
        1. Linear projection: [B, T] -> [B, T, d_model]
        2. Remove masked timesteps (encoder never sees masked data)
        3. Positional encoding
        4. FiLM-modulated Transformer layers (gamma, beta from static_attr)
        5. Normalize (NO POOLING - keeps sequence for CrossMAE decoder)
        6. Add static attributes as additional token
        7. Output sequence of visible tokens [B, L_visible+1, d_model]

    Args:
        in_feat: Input feature dimension (default: 1 for single-valued time series)
        stat_dim: Static attribute dimension (default: 11)
        d_model: Model dimension
        n_layers: Number of transformer layers
        nhead: Number of attention heads
        dropout: Dropout rate
        max_len: Maximum sequence length
        use_weighted_fm: Enable WeightedFeatureMaps (Phase 2)
        use_fm_layers: Which encoder layers to save
        use_input: Include input as layer 0
        use_spatial_agg: Enable spatial aggregation (NEW)
        spatial_agg_module: Pre-configured SpatialAggregation module (NEW)
    """

    def __init__(
        self,
        in_feat: int = 1,
        stat_dim: int = 11,
        d_model: int = 256,
        n_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        max_len: int = 90,
        use_weighted_fm: bool = False,  # Phase 2
        use_fm_layers: list = None,    # Phase 2
        use_input: bool = False,        # Phase 2
        use_spatial_agg: bool = False,  # NEW: Spatial aggregation
        spatial_agg_module: Optional[SpatialAggregation] = None,  # NEW
    ):
        super().__init__()

        self.in_feat = in_feat
        self.d_model = d_model
        self.use_weighted_fm = use_weighted_fm
        self.use_input = use_input
        self.use_spatial_agg = use_spatial_agg

        # Spatial aggregation (optional)
        if use_spatial_agg:
            assert spatial_agg_module is not None, \
                "spatial_agg_module must be provided when use_spatial_agg=True"
            self.spatial_agg = spatial_agg_module

            # Spatial positional embedding (learnable, shared across time)
            # Only for non-empty patches (类似image encoder只为valid patches创建)
            self.spatial_pos = nn.Parameter(
                torch.zeros(1, spatial_agg_module.num_non_empty_patches, d_model)
            )
            nn.init.normal_(self.spatial_pos, std=0.02)

        # Input projection
        self.in_proj = nn.Linear(in_feat, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # FiLM MLP layers (generate gamma and beta from static attributes)
        self.film_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(stat_dim, 2 * d_model),
                nn.Tanh()
            )
            for _ in range(n_layers)
        ])

        # FiLM-modulated transformer layers
        self.layers = nn.ModuleList([
            FiLMEncoderLayer(d_model, nhead, dropout)
            for _ in range(n_layers)
        ])

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

        # Static attribute projection (as additional token, not residual)
        self.attr_proj = nn.Linear(stat_dim, d_model)

        # Phase 2: Determine which layers to save for WeightedFeatureMaps
        if use_weighted_fm:
            if use_fm_layers is None:
                # Use all layers
                self.use_fm_layers = list(range(n_layers))
            else:
                # Use specified layers
                self.use_fm_layers = use_fm_layers
        else:
            self.use_fm_layers = []

    def forward(
        self,
        x_vec: Tensor,
        static_attr: Tensor,
        time_mask: Tensor
    ) -> tuple[Tensor, Dict]:
        """
        Forward pass

        Args:
            x_vec: [B, T] or [B, num_catchments, T] time series
            static_attr: [B, stat_dim] or [B, num_catchments, stat_dim] static attributes
            time_mask: [B, T] or [B, num_catchments, T] bool mask (True = masked, False = visible)

        Returns:
            encoder_output: [B, L_visible+1, d_model] - sequence of visible tokens + static token
            mask_info: dict with mask information for decoder
        """
        # ===== Step 0: Spatial Aggregation (if enabled) =====
        if self.use_spatial_agg:
            # Input: [B, num_catchments, T]
            # Output: [B, num_patches, T]
            assert x_vec.dim() == 3, \
                f"Expected [B, num_catchments, T], got {x_vec.shape}"

            B, num_catchments, T = x_vec.shape

            # Aggregate time series: [B, num_catchments, T] -> [B, num_patches, T]
            x_vec = self.spatial_agg(x_vec)  # [B, 100, T]

            # Aggregate static attributes: [B, num_catchments, stat_dim] -> [B, num_patches, stat_dim]
            static_attr = self.spatial_agg(static_attr)  # [B, 100, stat_dim]

            # Aggregate time mask using majority voting
            # If > 50% of catchments in a patch are masked, mark the patch as masked
            time_mask_float = time_mask.float()  # [B, num_catchments, T]
            time_mask_agg = self.spatial_agg(time_mask_float)  # [B, 100, T]
            time_mask = time_mask_agg > 0.5  # Majority vote: > 50% masked -> masked

            # 过滤empty patches，只保留non-empty patches (类似image encoder)
            non_empty_indices = self.spatial_agg.non_empty_patch_indices  # [64]
            x_vec = x_vec[:, non_empty_indices, :]  # [B, 64, T]
            static_attr = static_attr[:, non_empty_indices, :]  # [B, 64, stat_dim]
            time_mask = time_mask[:, non_empty_indices, :]  # [B, 64, T]

            # Now process as batch of patches
            # Reshape to [B*num_non_empty, T] to process each patch independently
            num_patches = len(non_empty_indices)  # 64
            x_vec = x_vec.reshape(B * num_patches, T)  # [B*64, T]
            static_attr = static_attr.reshape(B * num_patches, -1)  # [B*64, stat_dim]
            time_mask = time_mask.reshape(B * num_patches, T)  # [B*64, T]

            # Update effective batch size
            effective_B = B * num_patches

            # 记录spatial patch索引（用于后续添加spatial PE）
            # [0,1,...,63, 0,1,...,63] - 每个真实batch重复一次
            spatial_patch_indices = torch.arange(num_patches, device=x_vec.device).repeat(B)
        else:
            # Standard: [B, T] input
            effective_B = x_vec.shape[0]
            B, T = x_vec.shape
            num_patches = None
            spatial_patch_indices = None

        # Ensure static_attr has the same dtype as model weights (for mixed precision training)
        static_attr = static_attr.to(dtype=self.in_proj.weight.dtype)

        # ===== Step 1: Remove masked timesteps (MAE strategy) =====
        # Encoder should never see masked data
        visible_tokens_list = []
        lengths = []

        for b in range(effective_B):
            # Get visible timesteps for this sample
            visible_mask = ~time_mask[b]  # True = visible
            visible_data = x_vec[b, visible_mask]  # [num_visible]

            visible_tokens_list.append(visible_data)
            lengths.append(len(visible_data))

        # Pad to max length in batch
        max_len = max(lengths)
        x_padded = torch.zeros(effective_B, max_len, device=x_vec.device, dtype=self.in_proj.weight.dtype)

        for b in range(effective_B):
            x_padded[b, :lengths[b]] = visible_tokens_list[b].to(dtype=self.in_proj.weight.dtype)

        # Create padding mask for transformer (True = padding)
        padding_mask = torch.zeros(effective_B, max_len, device=x_vec.device, dtype=torch.bool)
        for b in range(effective_B):
            if lengths[b] < max_len:
                padding_mask[b, lengths[b]:] = True

        # ===== Step 2: Project to d_model =====
        # [B, max_len] -> [B, max_len, d_model]
        x = self.in_proj(x_padded.unsqueeze(-1))  # Add feature dim

        # ===== Step 3: Add positional encoding =====
        x = self.pos_enc(x)  # [effective_B, max_len, d_model] - Temporal PE

        # Add spatial PE (如果使用spatial aggregation)
        if self.use_spatial_agg:
            # 为每个样本添加对应的spatial PE
            # spatial_patch_indices: [B*num_patches] - 每个样本对应的patch索引(0-63)
            for b in range(effective_B):
                patch_idx = spatial_patch_indices[b]
                x[b, :, :] += self.spatial_pos[0, patch_idx]  # 添加spatial PE

        # ===== Step 4: FiLM-modulated Transformer layers =====
        if self.use_weighted_fm:
            # Phase 2: Collect multi-layer features
            x_feats = []

            # Optional: Include input as layer 0
            if self.use_input:
                x_feats.append(self.norm(x.clone()))

            # Process through FiLM layers
            for idx, (layer, film_mlp) in enumerate(zip(self.layers, self.film_mlps)):
                # Generate gamma and beta
                gb = film_mlp(static_attr)
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

                # Apply FiLM layer
                x = layer(x, gamma, beta, key_padding_mask=padding_mask)

                # Save specified layers
                if idx in self.use_fm_layers:
                    x_feats.append(self.norm(x.clone()))

            # Add static token to each feature map
            static_token = self.attr_proj(static_attr).unsqueeze(1)  # [effective_B, 1, d_model]
            x_feats_with_static = []
            for feat in x_feats:
                feat_with_static = torch.cat([feat, static_token], dim=1)  # [effective_B, L+1, d_model]
                x_feats_with_static.append(feat_with_static)

            # Update padding mask
            static_padding = torch.zeros(effective_B, 1, device=x_vec.device, dtype=torch.bool)
            padding_mask_full = torch.cat([padding_mask, static_padding], dim=1)

            # Reshape back if spatial aggregation was used
            if self.use_spatial_agg:
                # Reshape from [B*num_patches, L, d_model] to [B, num_patches, L, d_model]
                # Then process per-batch
                x_feats_reshaped = []
                for feat in x_feats_with_static:
                    _, L, D = feat.shape
                    feat_reshaped = feat.reshape(B, num_patches, L, D)
                    # For now, we'll keep the patch dimension separate
                    # The decoder needs to handle this appropriately
                    x_feats_reshaped.append(feat_reshaped)
                x_feats_with_static = x_feats_reshaped

                padding_mask_full = padding_mask_full.reshape(B, num_patches, -1)

            mask_info = {
                'mask': time_mask.reshape(B, num_patches, -1) if self.use_spatial_agg else time_mask,
                'lengths': lengths,
                'padding_mask': padding_mask_full,
                'use_spatial_agg': self.use_spatial_agg,
                'num_patches': num_patches if self.use_spatial_agg else None,
            }

            return x_feats_with_static, mask_info  # List of [B, (num_patches), L_visible+1, d_model]

        else:
            # Standard: Single layer output
            for layer, film_mlp in zip(self.layers, self.film_mlps):
                gb = film_mlp(static_attr)
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
                x = layer(x, gamma, beta, key_padding_mask=padding_mask)

            # ===== Step 5: Normalize (NO POOLING - CrossMAE style) =====
            x = self.norm(x)

            # ===== Step 6: Add static attributes as additional token =====
            static_token = self.attr_proj(static_attr).unsqueeze(1)
            encoder_output = torch.cat([x, static_token], dim=1)

            # Update padding mask
            static_padding = torch.zeros(effective_B, 1, device=x_vec.device, dtype=torch.bool)
            padding_mask_full = torch.cat([padding_mask, static_padding], dim=1)

            # Reshape back if spatial aggregation was used
            if self.use_spatial_agg:
                # Reshape from [B*num_patches, L, d_model] to [B, num_patches, L, d_model]
                _, L, D = encoder_output.shape
                encoder_output = encoder_output.reshape(B, num_patches, L, D)
                padding_mask_full = padding_mask_full.reshape(B, num_patches, -1)

            mask_info = {
                'mask': time_mask.reshape(B, num_patches, -1) if self.use_spatial_agg else time_mask,
                'lengths': lengths,
                'padding_mask': padding_mask_full,
                'use_spatial_agg': self.use_spatial_agg,
                'num_patches': num_patches if self.use_spatial_agg else None,
            }

            return encoder_output, mask_info  # [B, (num_patches), L_visible+1, d_model]


if __name__ == '__main__':
    """Unit test for VectorModalityEncoder"""

    print("=" * 60)
    print("Testing VectorModalityEncoder")
    print("=" * 60)

    # Create encoder
    encoder = VectorModalityEncoder(
        in_feat=1,
        stat_dim=11,
        d_model=256,
        n_layers=4,
        nhead=8,
        dropout=0.1,
        max_len=90,
    )

    # Test data
    B, T = 4, 90
    x_vec = torch.randn(B, T)  # [B, T] time series
    static_attr = torch.randn(B, 11)  # [B, 11] static attributes

    # Create mask (75% masked = True)
    time_mask = torch.rand(B, T) < 0.75  # [B, T] bool

    print(f"Input shape: {x_vec.shape}")
    print(f"Static attr shape: {static_attr.shape}")
    print(f"Mask shape: {time_mask.shape}")
    print(f"Mask ratio: {time_mask.float().mean().item():.2%}")

    # Forward pass
    token, mask_info = encoder(x_vec, static_attr, time_mask)

    print(f"\n✓ Output token shape: {token.shape}")
    print(f"✓ Mask info keys: {mask_info.keys()}")
    print(f"✓ Visible lengths: {mask_info['lengths']}")

    # Test backward
    loss = token.sum()
    loss.backward()

    print(f"✓ Backward pass successful")

    # Test with different mask ratios
    print(f"\n" + "=" * 60)
    print("Testing with different mask ratios")
    print("=" * 60)

    for mask_ratio in [0.0, 0.5, 0.75, 0.9]:
        time_mask = torch.rand(B, T) < mask_ratio
        token, mask_info = encoder(x_vec, static_attr, time_mask)
        visible_count = sum(mask_info['lengths'])
        print(f"Mask ratio {mask_ratio:.0%}: "
              f"Output shape {token.shape}, "
              f"Avg visible: {visible_count/B:.1f} timesteps")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
