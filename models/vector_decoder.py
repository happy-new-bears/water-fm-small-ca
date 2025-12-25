"""
Vector Modality Decoder (CrossMAE style)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from .layers import PositionalEncoding, CrossAttentionBlock, WeightedFeatureMaps
from .spatial_aggregation import SpatialAggregation


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
        spatial_agg_module: Optional[SpatialAggregation] = None,  # NEW: Spatial aggregation
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_time_steps = max_time_steps
        self.use_cross_attn = use_cross_attn
        self.use_weighted_fm = use_weighted_fm
        self.num_decoder_layers = num_decoder_layers
        self.spatial_agg = spatial_agg_module  # NEW: Store for reverse operation

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Temporal positional encoding
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

        # Prediction head
        self.pred_head = nn.Linear(decoder_dim, 1)  # Predict single value

    def forward(self, encoder_output, mask_info: Dict) -> Tensor:
        """
        Forward pass

        Args:
            encoder_output: Encoder features
                - If use_weighted_fm=False: [B, L_visible, encoder_dim]
                - If use_weighted_fm=True: list of [B, L_visible, encoder_dim]
            mask_info: dict with 'mask' [B, T], 'padding_mask' [B, L_visible]

        Returns:
            pred_vec: [B, T] - predicted time series
        """
        if self.use_cross_attn:
            return self._forward_cross_attn(encoder_output, mask_info)
        else:
            return self._forward_self_attn(encoder_output, mask_info)

    def _forward_cross_attn(self, encoder_output, mask_info: Dict) -> Tensor:
        """
        CrossMAE decoder: Only create queries for masked timesteps

        Phase 2: Support WeightedFeatureMaps
        - If use_weighted_fm=True, encoder_output is list of features
        - Use WeightedFeatureMaps to combine them
        - Each decoder layer uses different weighted combination

        Spatial Aggregation Support:
        - If use_spatial_agg=True, encoder_output has shape [B, num_patches, L, d_model]
        - Process each patch independently
        - Use spatial_agg.reverse() to aggregate predictions back to catchments
        """
        mask = mask_info['mask']  # [B, T] or [B, num_patches, T]
        padding_mask = mask_info.get('padding_mask')  # [B, L_visible] or [B, num_patches, L_visible]
        use_spatial_agg = mask_info.get('use_spatial_agg', False)
        num_patches = mask_info.get('num_patches', None)

        if use_spatial_agg:
            # Spatial aggregation mode
            # mask: [B, num_patches, T]
            # encoder_output: [B, num_patches, L, d_model] or list of such
            B_orig, num_patches, T = mask.shape

            # Reshape to process patches as batch
            # [B, num_patches, T] -> [B*num_patches, T]
            mask = mask.reshape(B_orig * num_patches, T)
            if padding_mask is not None:
                # [B, num_patches, L] -> [B*num_patches, L]
                padding_mask = padding_mask.reshape(B_orig * num_patches, -1)

            # Reshape encoder output
            if self.use_weighted_fm:
                # encoder_output is list of [B, num_patches, L, d_model]
                encoder_output_reshaped = []
                for feat in encoder_output:
                    B, P, L, D = feat.shape
                    feat_reshaped = feat.reshape(B * P, L, D)  # [B*num_patches, L, d_model]
                    encoder_output_reshaped.append(feat_reshaped)
                encoder_output = encoder_output_reshaped
            else:
                # encoder_output: [B, num_patches, L, d_model]
                B, P, L, D = encoder_output.shape
                encoder_output = encoder_output.reshape(B * P, L, D)  # [B*num_patches, L, d_model]

            B = B_orig * num_patches  # Effective batch size
        else:
            # Standard mode
            B, T = mask.shape
            B_orig = B
            num_patches = None

        # ===== Phase 2: Process encoder features =====
        if self.use_weighted_fm:
            # encoder_output is list of [B, L_visible, encoder_dim]
            assert isinstance(encoder_output, list), "Expected list of encoder features"

            # Combine multi-layer features: list -> [B, L, C, decoder_depth]
            weighted_features = self.weighted_fm(encoder_output)  # [B, L_visible, encoder_dim, num_decoder_layers]

            # Extract per-layer features (will use later in decoder loop)
            # Shape: [B, L_visible, encoder_dim, num_decoder_layers]
            encoder_features_per_layer = weighted_features
        else:
            # Standard: single encoder output
            # encoder_output: [B, L_visible, encoder_dim]
            encoder_features_per_layer = None

        # ===== Step 1: Create masked queries =====
        # Only for masked timesteps (TRUE = masked)
        masked_queries_list = []
        masked_positions_list = []  # Store (b, t) for reconstruction

        for b in range(B):
            for t in range(T):
                if mask[b, t]:  # True = masked
                    # Query = mask_token + temporal_pos
                    query = self.mask_token.squeeze(0).clone()  # [decoder_dim]
                    masked_queries_list.append(query)
                    masked_positions_list.append((b, t))

        if len(masked_queries_list) == 0:
            # Edge case: no masked timesteps
            return torch.zeros(B, T,
                             device=encoder_output.device if not isinstance(encoder_output, list) else encoder_output[0].device,
                             dtype=encoder_output.dtype if not isinstance(encoder_output, list) else encoder_output[0].dtype)

        # Stack queries: [total_masked, decoder_dim]
        queries = torch.stack(masked_queries_list, dim=0)

        # Add temporal positional encoding
        temporal_pe = self.temporal_pos.pe.squeeze(0)  # [max_time, decoder_dim]
        for i, (b, t) in enumerate(masked_positions_list):
            queries[i] = queries[i] + temporal_pe[t]

        queries = queries.unsqueeze(0)  # [1, total_masked, decoder_dim]

        # ===== Step 2: Process per batch =====
        all_predictions = []

        for b in range(B):
            # Get this batch's masked queries
            batch_mask_indices = [i for i, (bi, t) in enumerate(masked_positions_list) if bi == b]

            if len(batch_mask_indices) == 0:
                continue

            batch_queries = queries[0, batch_mask_indices, :].unsqueeze(0)  # [1, L_masked_b, D]

            # ===== Step 3: CrossAttention decoder with WeightedFeatureMaps =====
            x = batch_queries

            for layer_idx, blk in enumerate(self.decoder_blocks):
                # Get encoder features for this layer
                if self.use_weighted_fm:
                    # Extract this decoder layer's weighted feature map
                    # encoder_features_per_layer: [B, L_visible, encoder_dim, num_decoder_layers]
                    layer_features = encoder_features_per_layer[b:b+1, :, :, layer_idx]  # [1, L_visible, encoder_dim]

                    # Filter by padding mask
                    if padding_mask is not None:
                        valid_mask = ~padding_mask[b]  # [L_visible]
                        batch_encoder = layer_features[0:1, valid_mask, :]  # [1, L_valid, encoder_dim]
                    else:
                        batch_encoder = layer_features  # [1, L_visible, encoder_dim]

                    # Apply layer-wise normalization
                    batch_encoder = self.dec_norms[layer_idx](batch_encoder)
                else:
                    # Standard: use single encoder output
                    if padding_mask is not None:
                        valid_mask = ~padding_mask[b]  # [L_visible]
                        batch_encoder = encoder_output[b:b+1, valid_mask, :]  # [1, L_valid, encoder_dim]
                    else:
                        batch_encoder = encoder_output[b:b+1, :, :]  # [1, L_visible, encoder_dim]

                # Apply CrossAttention
                x = blk(x, batch_encoder)  # queries attend to this batch's visible tokens

            # ===== Step 4: Prediction =====
            x = self.decoder_norm(x)  # [1, L_masked_b, decoder_dim]
            predictions = self.pred_head(x).squeeze(-1)  # [1, L_masked_b]

            all_predictions.append((batch_mask_indices, predictions))

        # ===== Step 5: Reconstruct full output =====
        device = encoder_output[0].device if isinstance(encoder_output, list) else encoder_output.device
        dtype = encoder_output[0].dtype if isinstance(encoder_output, list) else encoder_output.dtype
        pred_vec = torch.zeros(B, T, device=device, dtype=dtype)

        for batch_mask_indices, predictions in all_predictions:
            for local_idx, global_idx in enumerate(batch_mask_indices):
                b, t = masked_positions_list[global_idx]
                pred_vec[b, t] = predictions[0, local_idx]

        # ===== Step 6: Reverse spatial aggregation (if enabled) =====
        if use_spatial_agg:
            # pred_vec is [B*num_patches, T] where num_patches = num_non_empty (e.g., 64)
            # Need to:
            # 1. Reshape to [B_orig, num_non_empty, T]
            # 2. Expand to [B_orig, num_patches_total, T] (e.g., 100) by inserting zeros for empty patches
            # 3. Reverse aggregation: [B_orig, num_patches_total, T] -> [B_orig, num_catchments, T]

            pred_vec = pred_vec.reshape(B_orig, num_patches, T)  # [B_orig, 64, T]

            # 扩展到完整的patch数量（包括empty patches）
            # 类似image decoder处理invalid patches
            num_patches_total = self.spatial_agg.num_patches  # 100
            pred_vec_full = torch.zeros(B_orig, num_patches_total, T,
                                       device=pred_vec.device, dtype=pred_vec.dtype)

            # 将non-empty patches的预测值填充到对应位置
            non_empty_indices = self.spatial_agg.non_empty_patch_indices  # [64]
            pred_vec_full[:, non_empty_indices, :] = pred_vec

            # Reverse aggregation: [B_orig, 100, T] -> [B_orig, num_catchments, T]
            pred_vec = self.spatial_agg.reverse(pred_vec_full)  # [B_orig, 604, T]

        return pred_vec

    def _forward_self_attn(self, encoder_output: Tensor, mask_info: Dict) -> Tensor:
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
