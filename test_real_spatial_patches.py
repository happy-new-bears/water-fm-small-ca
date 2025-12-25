"""
使用真实的spatial_patches_10x10.pt测试
（有36个empty patches的情况）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.spatial_aggregation import SpatialAggregation
from models.vector_encoder import VectorModalityEncoder
from models.vector_decoder import VectorModalityDecoder

print("=" * 80)
print("Testing with Real Spatial Patches (64 non-empty, 36 empty)")
print("=" * 80)

# 加载真实的spatial patches
spatial_data = torch.load('data/spatial_patches_10x10.pt')

patch_assignments = spatial_data['patch_assignments']
catchment_areas = spatial_data['catchment_areas']
num_catchments = spatial_data['num_catchments']
num_patches = spatial_data['num_patches']

print(f"\nReal Data Info:")
print(f"  Catchments: {num_catchments}")
print(f"  Total patches: {num_patches}")

# 创建SpatialAggregation
spatial_agg = SpatialAggregation(
    num_catchments=num_catchments,
    num_patches=num_patches,
    patch_assignments=patch_assignments,
    catchment_areas=catchment_areas,
)

patch_info = spatial_agg.get_patch_info()

print(f"\nPatch Statistics:")
print(f"  Non-empty patches: {spatial_agg.num_non_empty_patches}")
print(f"  Empty patches: {len(patch_info['empty_patches'])}")
print(f"  Empty patch IDs: {patch_info['empty_patches'][:10]}...")

# ========== Test with Real Data ==========
print(f"\n" + "=" * 80)
print("Test: Encoder/Decoder with 64 Non-empty Patches")
print("=" * 80)

encoder = VectorModalityEncoder(
    in_feat=1,
    stat_dim=11,
    d_model=256,
    n_layers=4,
    nhead=8,
    dropout=0.1,
    max_len=90,
    use_spatial_agg=True,
    spatial_agg_module=spatial_agg,
)

decoder = VectorModalityDecoder(
    encoder_dim=256,
    decoder_dim=128,
    max_time_steps=90,
    num_decoder_layers=4,
    nhead=8,
    dropout=0.1,
    use_cross_attn=True,
    spatial_agg_module=spatial_agg,
)

print(f"\nEncoder spatial_pos shape: {encoder.spatial_pos.shape}")
print(f"Expected: [1, {spatial_agg.num_non_empty_patches}, 256]")

# Test data
B, T = 2, 90
x_vec = torch.randn(B, num_catchments, T)
static_attr = torch.randn(B, num_catchments, 11)
time_mask = torch.rand(B, num_catchments, T) < 0.75

print(f"\nInput: [B={B}, catchments={num_catchments}, T={T}]")

# Forward
encoder_output, mask_info = encoder(x_vec, static_attr, time_mask)
print(f"Encoder output: {encoder_output.shape}")
print(f"Expected: [B={B}, non_empty={spatial_agg.num_non_empty_patches}, L_visible, D=256]")

assert encoder_output.shape[1] == spatial_agg.num_non_empty_patches, \
    f"Expected {spatial_agg.num_non_empty_patches} patches, got {encoder_output.shape[1]}"

pred_vec = decoder(encoder_output, mask_info)
print(f"Decoder output: {pred_vec.shape}")
print(f"Expected: [B={B}, catchments={num_catchments}, T={T}]")

assert pred_vec.shape == (B, num_catchments, T), \
    f"Expected shape ({B}, {num_catchments}, {T}), got {pred_vec.shape}"

# Backward
loss = pred_vec.sum()
loss.backward()

print(f"\n✓ All tests passed!")
print(f"✓ Encoder processes only {spatial_agg.num_non_empty_patches} non-empty patches")
print(f"✓ Decoder correctly outputs {num_catchments} catchments")
print(f"✓ Gradients flow correctly")

# Token reduction
tokens_original = num_catchments * T
tokens_processed = spatial_agg.num_non_empty_patches * T
reduction = (1 - tokens_processed / tokens_original) * 100

print(f"\nToken Reduction:")
print(f"  Original: {num_catchments} × {T} = {tokens_original}")
print(f"  Processed: {spatial_agg.num_non_empty_patches} × {T} = {tokens_processed}")
print(f"  Reduction: {reduction:.1f}%")

print(f"\n" + "=" * 80)
print("🎉 Real Data Test Passed!")
print("=" * 80)
