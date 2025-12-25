"""
Vector Modality Encoder with FiLM mechanism
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from .layers import PositionalEncoding, FiLMEncoderLayer


class VectorModalityEncoder(nn.Module):
    """
    Vector Modality Encoder with FiLM mechanism for static attribute fusion (CrossMAE style)

    Architecture:
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
        patch_size: int = 8,            # Spatial patch size for catchments
        modality_token: nn.Parameter = None,  # NEW: Modality token for cross-modal fusion
    ):
        super().__init__()

        self.in_feat = in_feat
        self.d_model = d_model
        self.use_weighted_fm = use_weighted_fm
        self.use_input = use_input
        self.patch_size = patch_size
        self.modality_token = modality_token  # Store modality token reference

        # Calculate number of patches (604 catchments / 8 = 76 patches)
        self.num_patches = 76  # This should match the data preprocessing

        # Input projection: project each catchment's time series independently
        # Then aggregate patch_size catchments into one token
        self.in_proj = nn.Linear(in_feat, d_model)

        # Spatial positional embedding (learnable, for all patches)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.normal_(self.spatial_pos, std=0.02)

        # Temporal positional encoding (fixed sincos)
        self.temporal_pos = PositionalEncoding(d_model, max_len)

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
        patch_mask: Tensor,
        catchment_padding_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Dict]:
        """
        Forward pass with spatial patchify

        Args:
            x_vec: [B, num_patches, patch_size, T] patch-level time series
            static_attr: [B, num_patches, patch_size, stat_dim] patch-level static attributes
            patch_mask: [B, num_patches, T] bool mask (True = masked PATCH, False = visible)
            catchment_padding_mask: [B, num_patches, patch_size] bool mask (True = padded catchment)

        Returns:
            encoder_output: [B, L_visible+1, d_model] - sequence of visible patch tokens + static token
            mask_info: dict with mask information for decoder
        """
        # Get dimensions
        B, num_patches, patch_size, T = x_vec.shape

        # Ensure dtype consistency
        static_attr = static_attr.to(dtype=self.in_proj.weight.dtype)

        # ===== Step 1: VECTORIZED visible patch selection (NO LOOPS!) =====
        # Encoder should never see masked patches
        # patch_mask: [B, num_patches, T] - a patch is visible if ALL timesteps are visible
        # Actually: a patch is visible if NONE of its timesteps are masked
        visible_patch_mask = ~patch_mask.any(dim=2)  # [B, num_patches] True = visible patch

        # Count visible patches per sample
        num_visible_per_sample = visible_patch_mask.sum(dim=1)  # [B]
        max_len = num_visible_per_sample.max().item()

        # Edge case: no visible patches
        if max_len == 0:
            # Return dummy output
            dummy_output = torch.zeros(B, 1, self.d_model, device=x_vec.device, dtype=self.in_proj.weight.dtype)
            dummy_padding_mask = torch.ones(B, 1, device=x_vec.device, dtype=torch.bool)
            mask_info = {
                'mask': patch_mask,
                'lengths': [0] * B,
                'padding_mask': dummy_padding_mask,
            }
            return dummy_output, mask_info

        # Check if all samples have same number of visible patches (should be true with fixed mask ratio)
        if (num_visible_per_sample == max_len).all():
            # FAST PATH: All samples have same length, VECTORIZED selection!
            # visible_patch_mask: [B, num_patches] boolean mask
            # x_vec: [B, num_patches, patch_size, T]

            # Directly use boolean indexing to select all visible patches (Flat)
            x_flat = x_vec[visible_patch_mask]  # [Total_Visible, patch_size, T]
            static_flat = static_attr[visible_patch_mask]  # [Total_Visible, patch_size, stat_dim]

            # Reshape back to [B, max_len, ...]
            # FAST PATH guarantees each sample has same number of visible patches
            x_padded = x_flat.view(B, max_len, patch_size, T)
            static_padded = static_flat.view(B, max_len, patch_size, -1)
            padding_mask = torch.zeros(B, max_len, device=x_vec.device, dtype=torch.bool)
            lengths = [max_len] * B

            # Also handle catchment_padding_mask (VECTORIZED!)
            if catchment_padding_mask is not None:
                catchment_padding_flat = catchment_padding_mask[visible_patch_mask]  # [Total_Visible, patch_size]
                catchment_padding_visible = catchment_padding_flat.view(B, max_len, patch_size)
            else:
                catchment_padding_visible = torch.zeros(B, max_len, patch_size, device=x_vec.device, dtype=torch.bool)

        else:
            # SLOW PATH: Different lengths, need padding
            x_padded = torch.zeros(B, max_len, patch_size, T, device=x_vec.device, dtype=self.in_proj.weight.dtype)
            static_padded = torch.zeros(B, max_len, patch_size, static_attr.shape[-1], device=x_vec.device, dtype=self.in_proj.weight.dtype)
            padding_mask = torch.zeros(B, max_len, device=x_vec.device, dtype=torch.bool)
            lengths = num_visible_per_sample.cpu().tolist()

            for b in range(B):
                length = lengths[b]
                x_padded[b, :length] = x_vec[b, visible_patch_mask[b]].to(dtype=self.in_proj.weight.dtype)
                static_padded[b, :length] = static_attr[b, visible_patch_mask[b]].to(dtype=self.in_proj.weight.dtype)
                if length < max_len:
                    padding_mask[b, length:] = True

            # Handle catchment_padding_mask
            if catchment_padding_mask is not None:
                catchment_padding_visible = []
                for b in range(B):
                    catch_pad_b = catchment_padding_mask[b, visible_patch_mask[b]]  # [num_visible, patch_size]
                    # Pad to max_len
                    if lengths[b] < max_len:
                        pad = torch.ones(max_len - lengths[b], patch_size, device=x_vec.device, dtype=torch.bool)
                        catch_pad_b = torch.cat([catch_pad_b, pad], dim=0)
                    catchment_padding_visible.append(catch_pad_b)
                catchment_padding_visible = torch.stack(catchment_padding_visible, dim=0)  # [B, max_len, patch_size]
            else:
                catchment_padding_visible = torch.zeros(B, max_len, patch_size, device=x_vec.device, dtype=torch.bool)

        # Aggregate: [B, max_len, patch_size, T] -> [B, max_len, T]
        # Mean pooling over catchments, excluding padded catchments
        # Use 1e-6 instead of 1e-8 for FP16 compatibility (1e-8 rounds to 0 in FP16)
        valid_mask = (~catchment_padding_visible).unsqueeze(-1).float()  # [B, max_len, patch_size, 1]
        x_aggregated = (x_padded * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-6)  # [B, max_len, T]

        # Similarly aggregate static attributes
        static_aggregated = (static_padded * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-6)  # [B, max_len, stat_dim]

        # ===== Step 3: Project to d_model =====
        # [B, max_len, T] -> [B, max_len, T, d_model]
        # Ensure dtype matches model weights for mixed precision training
        x_aggregated = x_aggregated.to(dtype=self.in_proj.weight.dtype)
        x = self.in_proj(x_aggregated.unsqueeze(-1))  # Add feature dim: [B, max_len, T, 1] -> [B, max_len, T, d_model]

        # ===== Step 4: Add positional embeddings (spatial + temporal) BEFORE flattening =====
        # Get visible patch indices
        # visible_patch_mask: [B, num_patches] - True for visible patches
        # We need to gather the patch indices for each sample
        patch_indices_per_sample = []
        for b in range(B):
            visible_idx = torch.nonzero(visible_patch_mask[b], as_tuple=True)[0]  # [max_len]
            patch_indices_per_sample.append(visible_idx)

        # Stack into [B, max_len]
        patch_indices = torch.stack(patch_indices_per_sample, dim=0)  # [B, max_len]

        # Gather spatial PE: self.spatial_pos is [1, num_patches, d_model]
        # spatial_emb: [B, max_len, d_model]
        spatial_emb = self.spatial_pos[0, patch_indices.view(-1)].view(B, max_len, self.d_model)

        # Add spatial PE (broadcast to time dimension)
        x = x + spatial_emb.unsqueeze(2)  # [B, max_len, T, d_model]

        # Gather temporal PE for each time step
        # temporal_pos.pe: [1, max_len, d_model] where max_len is max_time_steps
        temporal_pe = self.temporal_pos.pe.squeeze(0)  # [max_time_steps, d_model]
        temporal_emb = temporal_pe[:T, :]  # [T, d_model]

        # Add temporal PE (broadcast to batch and patch dimensions)
        x = x + temporal_emb.unsqueeze(0).unsqueeze(0)  # [B, max_len, T, d_model]

        # ===== Step 5: Flatten time dimension =====
        # [B, max_len, T, d_model] -> [B, max_len*T, d_model]
        x = x.reshape(B, max_len * T, self.d_model)

        # Add modality token (CAV-MAE style: after pos_embed, before transformer)
        if self.modality_token is not None:
            x = x + self.modality_token  # [1, 1, d_model] broadcast to [B, max_len*T, d_model]

        # Update padding mask for flattened sequence
        padding_mask_flat = padding_mask.unsqueeze(-1).expand(-1, -1, T).reshape(B, max_len * T)

        # ===== Step 6: FiLM-modulated Transformer layers =====
        # Use aggregated static attributes (pooled from patches)
        # static_aggregated: [B, max_len, stat_dim]
        # We need a single static vector per sample for FiLM
        # Pool over patches (use 1e-6 for FP16 compatibility)
        static_pooled = (static_aggregated * (~padding_mask).unsqueeze(-1).float()).sum(dim=1) / ((~padding_mask).sum(dim=1, keepdim=True).float() + 1e-6)  # [B, stat_dim]
        # Ensure dtype consistency for mixed precision
        static_pooled = static_pooled.to(dtype=self.in_proj.weight.dtype)

        if self.use_weighted_fm:
            # Phase 2: Collect multi-layer features
            x_feats = []

            # Optional: Include input as layer 0
            if self.use_input:
                x_feats.append(self.norm(x.clone()))

            # Process through FiLM layers
            for idx, (layer, film_mlp) in enumerate(zip(self.layers, self.film_mlps)):
                # Generate gamma and beta from pooled static attributes
                gb = film_mlp(static_pooled)
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

                # Apply FiLM layer
                x = layer(x, gamma, beta, key_padding_mask=padding_mask_flat)

                # Save specified layers
                if idx in self.use_fm_layers:
                    x_feats.append(self.norm(x.clone()))

            # Add static token to each feature map
            static_token = self.attr_proj(static_pooled).unsqueeze(1)  # [B, 1, d_model]
            x_feats_with_static = []
            for feat in x_feats:
                feat_with_static = torch.cat([feat, static_token], dim=1)  # [B, L+1, d_model]
                x_feats_with_static.append(feat_with_static)

            # Update padding mask
            static_padding = torch.zeros(B, 1, device=x_vec.device, dtype=torch.bool)
            padding_mask_full = torch.cat([padding_mask_flat, static_padding], dim=1)

            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask_full,
            }

            return x_feats_with_static, mask_info  # List of [B, L_visible+1, d_model]

        else:
            # Standard: Single layer output
            for layer, film_mlp in zip(self.layers, self.film_mlps):
                gb = film_mlp(static_pooled)
                gamma, beta = gb.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
                x = layer(x, gamma, beta, key_padding_mask=padding_mask_flat)

            # ===== Step 7: Normalize (NO POOLING - CrossMAE style) =====
            x = self.norm(x)

            # ===== Step 8: Add static attributes as additional token =====
            static_token = self.attr_proj(static_pooled).unsqueeze(1)
            encoder_output = torch.cat([x, static_token], dim=1)

            # Update padding mask
            static_padding = torch.zeros(B, 1, device=x_vec.device, dtype=torch.bool)
            padding_mask_full = torch.cat([padding_mask_flat, static_padding], dim=1)

            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask_full,
            }

            return encoder_output, mask_info  # [B, L_visible+1, d_model]


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
