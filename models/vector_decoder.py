"""
Vector Modality Decoder (CrossMAE style)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from .layers import PositionalEncoding, CrossAttentionBlock, WeightedFeatureMaps


class VectorModalityDecoder(nn.Module):
    """
    Vector Modality Decoder for MAE reconstruction (CrossMAE Architecture)

    Architecture (Phase 1 - Complete CrossMAE):
        1. Accept encoder sequence [B, L_visible, d_model]
        2. Create masked queries (only for masked timesteps)
        3. Add temporal position embeddings to queries
        4. CrossAttention decoder: queries attend to encoder sequence
        5. Linear head to predict values

    Phase 2 Enhancement (WeightedFeatureMaps):
        - Accept list of encoder features from multiple layers
        - Learn weighted combinations for each decoder layer
        - Each decoder layer uses a different weighted feature map

    Similar to Image Decoder but simpler (1D temporal only, no spatial dimension)

    Args:
        encoder_dim: Encoder output dimension
        decoder_dim: Decoder embedding dimension
        max_time_steps: Maximum sequence length
        num_decoder_layers: Number of transformer layers
        nhead: Number of attention heads
        dropout: Dropout rate
        use_cross_attn: Use CrossAttention (default: True)
        decoder_self_attn: Use self-attention in decoder (default: False)
        use_weighted_fm: Use WeightedFeatureMaps (Phase 2, default: False)
        num_encoder_layers: Number of encoder layers for WeightedFeatureMaps
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_dim: int = 128,
        max_time_steps: int = 90,
        num_decoder_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        decoder_self_attn: bool = False,
        use_weighted_fm: bool = False,  # Phase 2
        num_encoder_layers: int = 4,    # Phase 2
        patch_size: int = 8,            # Spatial patch size for catchments
        num_catchments: int = 604,      # Total number of catchments
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_time_steps = max_time_steps
        self.use_cross_attn = use_cross_attn
        self.use_weighted_fm = use_weighted_fm
        self.num_decoder_layers = num_decoder_layers
        self.patch_size = patch_size
        self.num_catchments = num_catchments

        # Calculate number of patches (604 catchments / 8 = 76 patches)
        self.num_patches = 76  # This should match the encoder

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Spatial positional embedding (learnable, for all patches)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )
        nn.init.normal_(self.spatial_pos, std=0.02)

        # Temporal positional encoding (fixed sincos)
        self.temporal_pos = PositionalEncoding(decoder_dim, max_time_steps)

        # Decoder blocks
        if use_cross_attn:
            # CrossAttention decoder (CrossMAE)
            self.decoder_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_dim,
                    num_heads=nhead,
                    mlp_ratio=4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    self_attn=decoder_self_attn,
                )
                for _ in range(num_decoder_layers)
            ])
            self.decoder_norm = nn.LayerNorm(decoder_dim)

            # Phase 2: WeightedFeatureMaps
            if use_weighted_fm:
                # WeightedFeatureMaps module
                self.weighted_fm = WeightedFeatureMaps(
                    num_layers=num_encoder_layers,
                    embed_dim=encoder_dim,
                    decoder_depth=num_decoder_layers,
                )

                # Layer-wise normalization (one for each decoder layer)
                self.dec_norms = nn.ModuleList([
                    nn.LayerNorm(encoder_dim)
                    for _ in range(num_decoder_layers)
                ])
        else:
            # Fallback: Self-attention decoder (standard MAE)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=nhead,
                dim_feedforward=4 * decoder_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                decoder_layer,
                num_layers=num_decoder_layers,
            )
            self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Prediction head (predicts for each catchment in patch)
        self.pred_head = nn.Linear(decoder_dim, patch_size)  # Predict patch_size values

    def forward(self, encoder_output, mask_info: Dict, decoder_modality_token=None, key_padding_mask=None) -> Tensor:
        """
        Forward pass with patch-level mask

        Args:
            encoder_output: Encoder features
                - If use_weighted_fm=False: [B, L_visible, encoder_dim]
                - If use_weighted_fm=True: list of [B, L_visible, encoder_dim]
            mask_info: dict with 'mask' [B, num_patches, T], 'padding_mask' [B, L_visible]
            decoder_modality_token: [1, 1, decoder_dim] decoder modality token (optional)
            key_padding_mask: [B, L_total] - Global padding mask from all modalities (True = padded)

        Returns:
            pred_vec: [B, num_catchments, T] - predicted time series for all catchments
        """
        if self.use_cross_attn:
            return self._forward_cross_attn(encoder_output, mask_info, decoder_modality_token, key_padding_mask)
        else:
            return self._forward_self_attn(encoder_output, mask_info, decoder_modality_token)

    def _forward_cross_attn(self, encoder_output, mask_info: Dict, decoder_modality_token=None, key_padding_mask=None) -> Tensor:
        """
        Vectorized CrossMAE decoder for vectors (NO LOOPS over batch!)

        Key optimization: Process entire batch in parallel instead of per-sample loops.
        Assumes fixed mask ratio, so all samples have same number of masked patches.

        Args:
            key_padding_mask: [B, L_total] - Global padding mask from all modalities (True = padded)
        """
        patch_mask = mask_info['mask']  # [B, num_patches, T]
        padding_mask_local = mask_info.get('padding_mask')  # [B, L_visible] - local mask from this modality
        B, num_patches, T = patch_mask.shape

        # Use global padding mask if provided, otherwise fall back to local
        # IMPORTANT: key_padding_mask should match encoder_output length (which is fused_features in MultiModalMAE)
        if key_padding_mask is None:
            key_padding_mask = padding_mask_local

        # Calculate actual number of catchments (accounting for padding)
        num_padded = num_patches * self.patch_size
        num_actual = self.num_catchments

        # ===== Phase 2: Process encoder features =====
        if self.use_weighted_fm:
            # encoder_output is list of [B, L_visible, encoder_dim]
            assert isinstance(encoder_output, list), "Expected list of encoder features"
            # Combine multi-layer features: list -> [B, L, C, decoder_depth]
            weighted_features = self.weighted_fm(encoder_output)  # [B, L_visible, encoder_dim, num_decoder_layers]
            encoder_features_per_layer = weighted_features
        else:
            # Standard: single encoder output [B, L_visible, encoder_dim]
            encoder_features_per_layer = None

        # ===== Step 1: VECTORIZED Query Creation (handle variable-length masked sequences) =====
        # Calculate number of masked positions per sample
        num_masked_per_sample = patch_mask.sum(dim=(1, 2))  # [B]
        max_masked = num_masked_per_sample.max().item()

        # Get all masked positions
        # nonzero() returns indices in order (b, p, t), which we rely on
        indices = patch_mask.nonzero(as_tuple=False)  # [Total_Masked, 3]

        # Extract p and t indices (flattened 1D tensors)
        p_indices_flat = indices[:, 1]  # [Total_Masked]
        t_indices_flat = indices[:, 2]  # [Total_Masked]

        # Check if all samples have same number of masked positions
        if (num_masked_per_sample == max_masked).all():
            # FAST PATH: All samples have same length, no padding needed
            p_indices = p_indices_flat.view(B, max_masked)  # [B, k]
            t_indices = t_indices_flat.view(B, max_masked)  # [B, k]
        else:
            # SLOW PATH: Different lengths, need padding
            p_indices = torch.zeros(B, max_masked, device=patch_mask.device, dtype=torch.long)
            t_indices = torch.zeros(B, max_masked, device=patch_mask.device, dtype=torch.long)

            offset = 0
            for b in range(B):
                length = num_masked_per_sample[b].item()
                p_indices[b, :length] = p_indices_flat[offset:offset+length]
                t_indices[b, :length] = t_indices_flat[offset:offset+length]
                # Padded positions will have index 0 (dummy values)
                offset += length

        # Create Queries [B, k, decoder_dim]
        queries = self.mask_token.expand(B, max_masked, -1).clone()

        # Add Spatial PE (Gathering, NO LOOP!)
        # self.spatial_pos: [1, num_patches, decoder_dim]
        # p_indices: [B, k] -> Gather -> [B, k, decoder_dim]
        spatial_emb = self.spatial_pos[0, p_indices]  # [B, k, decoder_dim]
        queries = queries + spatial_emb

        # Add Temporal PE (Gathering, NO LOOP!)
        # temporal_pos.pe: [1, max_len, decoder_dim] -> [max_len, decoder_dim]
        temporal_emb = self.temporal_pos.pe.squeeze(0)[t_indices]  # [B, k, decoder_dim]
        queries = queries + temporal_emb

        # Add Decoder Modality Token (CAV-MAE style: after pos_embed)
        if decoder_modality_token is not None:
            queries = queries + decoder_modality_token  # [1, 1, decoder_dim] broadcast to [B, k, decoder_dim]

        # ===== Step 2: Batched Cross Attention (PARALLEL!) =====
        x = queries  # [B, k, decoder_dim]

        for layer_idx, blk in enumerate(self.decoder_blocks):
            # Get Encoder Features
            if self.use_weighted_fm:
                # [B, L_visible, encoder_dim, Layers] -> [B, L_visible, encoder_dim]
                batch_encoder = encoder_features_per_layer[:, :, :, layer_idx]

                # Apply Layer-wise Norm
                batch_encoder = self.dec_norms[layer_idx](batch_encoder)
            else:
                # Standard
                if isinstance(encoder_output, list):
                    batch_encoder = encoder_output[-1]  # Use last layer
                else:
                    batch_encoder = encoder_output  # [B, L_visible, encoder_dim]

            # CrossAttention now processes ENTIRE BATCH in parallel!
            # Pass key_padding_mask to prevent attention to padded positions
            x = blk(x, batch_encoder, key_padding_mask=key_padding_mask)  # [B, k, decoder_dim]

        # ===== Step 3: Prediction =====
        x = self.decoder_norm(x)  # [B, k, decoder_dim]
        predictions_patch = self.pred_head(x)  # [B, k, patch_size]

        # ===== Step 4: Scatter back & Unpatchify (handle variable-length predictions) =====
        device = predictions_patch.device
        dtype = predictions_patch.dtype

        # Target: [B, num_patches, T, patch_size]
        # This shape makes it easier to assign using boolean indexing
        pred_grid = torch.zeros(B, num_patches, T, self.patch_size, device=device, dtype=dtype)

        # Handle variable-length predictions
        if (num_masked_per_sample == max_masked).all():
            # FAST PATH: All samples have same length, direct reshape
            # predictions_patch is [B, k, patch_size] -> flatten -> [Total_Masked, patch_size]
            pred_grid[patch_mask] = predictions_patch.reshape(-1, self.patch_size)
        else:
            # SLOW PATH: Different lengths, need to handle per-sample
            # Extract only valid predictions (exclude padding)
            offset = 0
            all_preds = []
            for b in range(B):
                length = num_masked_per_sample[b].item()
                # predictions_patch[b, :length] are valid predictions
                valid_preds = predictions_patch[b, :length]  # [length, patch_size]
                all_preds.append(valid_preds)
                offset += length

            # Concatenate all valid predictions
            all_preds_cat = torch.cat(all_preds, dim=0)  # [Total_Masked_Actual, patch_size]

            # Scatter to grid
            pred_grid[patch_mask] = all_preds_cat

        # Now convert back to [B, num_catchments, T]
        # [B, num_patches, T, patch_size] -> [B, num_patches, patch_size, T]
        pred_vec_padded = pred_grid.permute(0, 1, 3, 2).reshape(B, num_padded, T)

        # Crop padding: [B, num_padded, T] -> [B, num_actual, T]
        pred_vec = pred_vec_padded[:, :num_actual, :]

        return pred_vec

    def _forward_self_attn(self, encoder_output: Tensor, mask_info: Dict, decoder_modality_token=None) -> Tensor:
        """
        Fallback: Standard self-attention decoder (for compatibility)
        """
        mask = mask_info['mask']  # [B, T]
        padding_mask = mask_info.get('padding_mask')  # [B, L_visible]
        B, T = mask.shape

        # Pool encoder sequence to single token
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()  # [B, L_visible, 1]
            encoder_token = (encoder_output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            encoder_token = encoder_output.mean(dim=1)  # [B, encoder_dim]

        # Create full sequence with mask tokens
        x = self.mask_token.expand(B, T, -1).clone()  # [B, T, decoder_dim]

        # Fill visible positions with pooled encoder token
        for b in range(B):
            visible_mask = ~mask[b]  # [T]
            if visible_mask.any():
                x[b, visible_mask] = encoder_token[b].unsqueeze(0)

        # Add temporal PE
        x = self.temporal_pos(x)  # [B, T, decoder_dim]

        # Add Decoder Modality Token (if provided)
        if decoder_modality_token is not None:
            x = x + decoder_modality_token  # [1, 1, decoder_dim] broadcast

        # Self-attention transformer
        x = self.transformer(x)
        x = self.decoder_norm(x)

        # Prediction
        pred_vec = self.pred_head(x).squeeze(-1)  # [B, T]

        return pred_vec


if __name__ == '__main__':
    """Unit test for VectorModalityDecoder"""

    print("=" * 60)
    print("Testing VectorModalityDecoder")
    print("=" * 60)

    # Create decoder
    decoder = VectorModalityDecoder(
        encoder_dim=256,
        decoder_dim=128,
        max_time_steps=90,
        num_decoder_layers=4,
        nhead=8,
        dropout=0.1,
    )

    # Test data
    B, T = 4, 90
    encoder_token = torch.randn(B, 256)  # [B, encoder_dim]

    # Create mask (75% masked)
    mask = torch.rand(B, T) < 0.75  # [B, T] bool

    mask_info = {'mask': mask}

    print(f"Encoder token shape: {encoder_token.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.float().mean().item():.2%}")

    # Forward pass
    pred_vec = decoder(encoder_token, mask_info)

    print(f"\n✓ Predicted vector shape: {pred_vec.shape}")
    assert pred_vec.shape == (B, T), f"Expected shape ({B}, {T}), got {pred_vec.shape}"

    # Test backward
    loss = pred_vec.sum()
    loss.backward()

    print(f"✓ Backward pass successful")

    # Test reconstruction loss
    target_vec = torch.randn(B, T)
    mse_loss = torch.nn.functional.mse_loss(
        pred_vec, target_vec, reduction='none'
    )  # [B, T]

    # Only compute loss on masked positions
    masked_loss = (mse_loss * mask.float()).sum() / mask.sum()

    print(f"✓ Masked reconstruction loss: {masked_loss.item():.4f}")

    # Test with different mask ratios
    print(f"\n" + "=" * 60)
    print("Testing with different mask ratios")
    print("=" * 60)

    for mask_ratio in [0.0, 0.5, 0.75, 0.9]:
        mask = torch.rand(B, T) < mask_ratio
        mask_info = {'mask': mask}
        pred_vec = decoder(encoder_token, mask_info)
        print(f"Mask ratio {mask_ratio:.0%}: "
              f"Output shape {pred_vec.shape}, "
              f"Num masked: {mask.sum().item()}/{B*T}")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
