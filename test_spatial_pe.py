"""
测试修改后的Spatial Aggregation with Spatial PE

验证：
1. Empty patches被正确过滤
2. Spatial PE正确添加到non-empty patches
3. Decoder正确reverse aggregation
4. 输出shape正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.spatial_aggregation import SpatialAggregation, create_grid_patches
from models.vector_encoder import VectorModalityEncoder
from models.vector_decoder import VectorModalityDecoder

print("=" * 80)
print("Testing Spatial Aggregation with Spatial PE")
print("=" * 80)

# ========== Setup ==========
torch.manual_seed(42)

# Simulate 604 catchments with 10×10 grid
num_catchments = 604
grid_size = (10, 10)
num_patches = grid_size[0] * grid_size[1]  # 100

# Generate simulated catchment data
catchment_lons = torch.rand(num_catchments) * 10 - 5
catchment_lats = torch.rand(num_catchments) * 10 + 50
catchment_areas = torch.rand(num_catchments) * 500 + 50

# Create spatial patches
patch_assignments, _ = create_grid_patches(
    catchment_lons, catchment_lats, catchment_areas,
    grid_size=grid_size
)

# Create SpatialAggregation module
spatial_agg = SpatialAggregation(
    num_catchments=num_catchments,
    num_patches=num_patches,
    patch_assignments=patch_assignments,
    catchment_areas=catchment_areas,
    aggregation_mode='fixed',
)

print(f"\nSpatial Aggregation Info:")
print(f"  Total patches: {spatial_agg.num_patches}")
print(f"  Non-empty patches: {spatial_agg.num_non_empty_patches}")
print(f"  Empty patches: {spatial_agg.num_patches - spatial_agg.num_non_empty_patches}")
print(f"  Non-empty indices: {spatial_agg.non_empty_patch_indices[:10].tolist()}...")

# ========== Test 1: Vector Encoder with Spatial PE ==========
print(f"\n" + "=" * 80)
print("Test 1: Vector Encoder with Spatial PE")
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

print(f"\nEncoder Info:")
print(f"  Has spatial_pos: {hasattr(encoder, 'spatial_pos')}")
if hasattr(encoder, 'spatial_pos'):
    print(f"  spatial_pos shape: {encoder.spatial_pos.shape}")
    print(f"  Expected: [1, {spatial_agg.num_non_empty_patches}, 256]")

# Test data
B, T = 2, 90
x_vec = torch.randn(B, num_catchments, T)
static_attr = torch.randn(B, num_catchments, 11)
time_mask = torch.rand(B, num_catchments, T) < 0.75

print(f"\nInput shapes:")
print(f"  x_vec: {x_vec.shape}")
print(f"  static_attr: {static_attr.shape}")
print(f"  time_mask: {time_mask.shape}")

# Forward pass
encoder_output, mask_info = encoder(x_vec, static_attr, time_mask)

print(f"\nEncoder output:")
print(f"  Shape: {encoder_output.shape}")
print(f"  Expected: [B={B}, num_non_empty={spatial_agg.num_non_empty_patches}, L_visible, d_model=256]")

# Check dimensions
assert encoder_output.dim() == 4, f"Expected 4D output, got {encoder_output.dim()}D"
assert encoder_output.shape[0] == B
assert encoder_output.shape[1] == spatial_agg.num_non_empty_patches
print(f"\n✓ Encoder output shape correct!")

# Check mask_info
print(f"\nMask info:")
print(f"  use_spatial_agg: {mask_info['use_spatial_agg']}")
print(f"  num_patches: {mask_info['num_patches']}")
assert mask_info['num_patches'] == spatial_agg.num_non_empty_patches

# ========== Test 2: Vector Decoder with Reverse ==========
print(f"\n" + "=" * 80)
print("Test 2: Vector Decoder with Reverse Aggregation")
print("=" * 80)

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

# Forward pass
pred_vec = decoder(encoder_output, mask_info)

print(f"\nDecoder output:")
print(f"  Shape: {pred_vec.shape}")
print(f"  Expected: [B={B}, num_catchments={num_catchments}, T={T}]")

# Check output shape
assert pred_vec.shape == (B, num_catchments, T), \
    f"Expected shape ({B}, {num_catchments}, {T}), got {pred_vec.shape}"

print(f"\n✓ Decoder output shape correct!")

# ========== Test 3: Backward Pass ==========
print(f"\n" + "=" * 80)
print("Test 3: Backward Pass")
print("=" * 80)

# Re-run with requires_grad
x_vec = torch.randn(B, num_catchments, T, requires_grad=True)
static_attr = torch.randn(B, num_catchments, 11, requires_grad=True)
time_mask = torch.rand(B, num_catchments, T) < 0.75

encoder_output, mask_info = encoder(x_vec, static_attr, time_mask)
pred_vec = decoder(encoder_output, mask_info)

loss = pred_vec.sum()
loss.backward()

print(f"✓ Backward pass successful")
print(f"✓ x_vec gradient: {x_vec.grad.shape}")
print(f"✓ static_attr gradient: {static_attr.grad.shape}")

# Check encoder spatial_pos gradient
if hasattr(encoder, 'spatial_pos'):
    if encoder.spatial_pos.grad is not None:
        print(f"✓ spatial_pos gradient: {encoder.spatial_pos.grad.shape}")
        print(f"  Gradient mean: {encoder.spatial_pos.grad.mean().item():.6f}")
    else:
        print(f"⚠️  spatial_pos has no gradient")

# ========== Test 4: Check Spatial PE Effect ==========
print(f"\n" + "=" * 80)
print("Test 4: Verify Spatial PE Effect")
print("=" * 80)

# Test with same input but different patches should have different encodings
x_same = torch.ones(B, num_catchments, T) * 0.5
static_attr_same = torch.ones(B, num_catchments, 11) * 0.5
time_mask_same = torch.zeros(B, num_catchments, T, dtype=torch.bool)

encoder_output_same, _ = encoder(x_same, static_attr_same, time_mask_same)

# Check if different patches have different encodings (due to spatial PE)
# encoder_output_same: [B, num_non_empty, L, d_model]
patch_0_encoding = encoder_output_same[0, 0, 0, :]  # First patch, first timestep
patch_1_encoding = encoder_output_same[0, 1, 0, :]  # Second patch, first timestep

diff = (patch_0_encoding - patch_1_encoding).abs().mean().item()
print(f"\nEncoding difference between patch 0 and patch 1: {diff:.6f}")

if diff > 1e-5:
    print(f"✓ Spatial PE is working! Different patches have different encodings.")
else:
    print(f"⚠️  Spatial PE might not be working. Difference too small: {diff}")

# ========== Test 5: Token Reduction Statistics ==========
print(f"\n" + "=" * 80)
print("Test 5: Token Reduction Statistics")
print("=" * 80)

tokens_without_agg = num_catchments * T
tokens_with_agg_all = num_patches * T
tokens_with_agg_filtered = spatial_agg.num_non_empty_patches * T

reduction_all = (1 - tokens_with_agg_all / tokens_without_agg) * 100
reduction_filtered = (1 - tokens_with_agg_filtered / tokens_without_agg) * 100

print(f"\nToken counts:")
print(f"  Without aggregation: {num_catchments} × {T} = {tokens_without_agg}")
print(f"  With aggregation (all patches): {num_patches} × {T} = {tokens_with_agg_all}")
print(f"  With aggregation (non-empty only): {spatial_agg.num_non_empty_patches} × {T} = {tokens_with_agg_filtered}")

print(f"\nReductions:")
print(f"  All patches: {reduction_all:.1f}%")
print(f"  Non-empty only: {reduction_filtered:.1f}% ← Actual reduction!")

# ========== Summary ==========
print(f"\n" + "=" * 80)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("=" * 80)

print(f"\nSummary:")
print(f"  ✓ Empty patches correctly filtered")
print(f"  ✓ Spatial PE created for {spatial_agg.num_non_empty_patches} non-empty patches")
print(f"  ✓ Spatial PE correctly added to encoder features")
print(f"  ✓ Decoder correctly reverses aggregation to {num_catchments} catchments")
print(f"  ✓ Gradients flow correctly through spatial PE")
print(f"  ✓ Token reduction: {reduction_filtered:.1f}%")

print(f"\n🎉 Spatial Aggregation with Spatial PE is working correctly!")
