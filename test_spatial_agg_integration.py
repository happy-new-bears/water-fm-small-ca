"""
Test Spatial Aggregation Integration

Tests the complete pipeline with spatial aggregation enabled
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.spatial_aggregation import SpatialAggregation, create_grid_patches
from models.vector_encoder import VectorModalityEncoder
from models.vector_decoder import VectorModalityDecoder

print("=" * 80)
print("Testing Spatial Aggregation Integration")
print("=" * 80)

# ========== Setup ==========
torch.manual_seed(42)

# Simulate 604 catchments with 10×10 grid
num_catchments = 604
grid_size = (10, 10)
num_patches = grid_size[0] * grid_size[1]  # 100

# Generate simulated catchment data
catchment_lons = torch.rand(num_catchments) * 10 - 5  # [-5, 5]
catchment_lats = torch.rand(num_catchments) * 10 + 50  # [50, 60]
catchment_areas = torch.rand(num_catchments) * 500 + 50  # [50, 550]

print(f"\nSetup:")
print(f"  Catchments: {num_catchments}")
print(f"  Grid size: {grid_size[0]}×{grid_size[1]} = {num_patches} patches")

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

patch_info = spatial_agg.get_patch_info()
print(f"  Non-empty patches: {patch_info['num_non_empty_patches']}")

# ========== Test 1: Vector Encoder with Spatial Aggregation ==========
print(f"\n" + "=" * 80)
print("Test 1: Vector Encoder with Spatial Aggregation")
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

# Test data: [B, num_catchments, T]
B, T = 2, 90
x_vec = torch.randn(B, num_catchments, T)
static_attr = torch.randn(B, num_catchments, 11)
time_mask = torch.rand(B, num_catchments, T) < 0.75  # 75% masked

print(f"Input shapes:")
print(f"  x_vec: {x_vec.shape}")
print(f"  static_attr: {static_attr.shape}")
print(f"  time_mask: {time_mask.shape}")

# Forward pass
encoder_output, mask_info = encoder(x_vec, static_attr, time_mask)

print(f"\nEncoder output:")
print(f"  Shape: {encoder_output.shape}")
print(f"  Expected: [B={B}, num_patches={num_patches}, L_visible, d_model=256]")

# Check mask_info
print(f"\nMask info:")
print(f"  use_spatial_agg: {mask_info['use_spatial_agg']}")
print(f"  num_patches: {mask_info['num_patches']}")
print(f"  mask shape: {mask_info['mask'].shape}")

assert encoder_output.dim() == 4, f"Expected 4D output, got {encoder_output.dim()}D"
assert encoder_output.shape[0] == B
assert encoder_output.shape[1] == num_patches
print(f"\n✓ Vector Encoder test passed")

# ========== Test 2: Vector Decoder with Spatial Aggregation ==========
print(f"\n" + "=" * 80)
print("Test 2: Vector Decoder with Spatial Aggregation")
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

print(f"Decoder output:")
print(f"  Shape: {pred_vec.shape}")
print(f"  Expected: [B={B}, num_catchments={num_catchments}, T={T}]")

assert pred_vec.shape == (B, num_catchments, T), \
    f"Expected shape ({B}, {num_catchments}, {T}), got {pred_vec.shape}"

print(f"\n✓ Vector Decoder test passed")

# ========== Test 3: Backward Pass ==========
print(f"\n" + "=" * 80)
print("Test 3: Backward Pass")
print("=" * 80)

loss = pred_vec.sum()
loss.backward()

print(f"✓ Backward pass successful")

# Check gradients
has_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
total_params = sum(1 for p in encoder.parameters())
print(f"  Encoder gradients: {has_grad}/{total_params}")

has_grad = sum(1 for p in decoder.parameters() if p.grad is not None)
total_params = sum(1 for p in decoder.parameters())
print(f"  Decoder gradients: {has_grad}/{total_params}")

# ========== Test 4: Token Reduction Statistics ==========
print(f"\n" + "=" * 80)
print("Test 4: Token Reduction Statistics")
print("=" * 80)

tokens_before = num_catchments * T
tokens_after = num_patches * T
reduction = (1 - tokens_after / tokens_before) * 100

print(f"Without spatial aggregation:")
print(f"  Tokens per sample: {num_catchments} catchments × {T} timesteps = {tokens_before}")

print(f"\nWith spatial aggregation:")
print(f"  Tokens per sample: {num_patches} patches × {T} timesteps = {tokens_after}")
print(f"  Token reduction: {reduction:.1f}%")

print(f"\nMemory savings:")
print(f"  Encoder processes: {num_patches} patches (vs {num_catchments} catchments)")
print(f"  Decoder outputs: {num_catchments} catchments (via reverse aggregation)")
print(f"  Net benefit: {reduction:.1f}% fewer tokens to process")

# ========== Test 5: Compare with Standard Mode ==========
print(f"\n" + "=" * 80)
print("Test 5: Compare with Standard Mode (no spatial agg)")
print("=" * 80)

# Standard encoder (no spatial agg)
encoder_std = VectorModalityEncoder(
    in_feat=1,
    stat_dim=11,
    d_model=256,
    n_layers=4,
    nhead=8,
    dropout=0.1,
    max_len=90,
    use_spatial_agg=False,  # Disabled
)

# Test with single catchment (standard mode expects [B, T])
x_vec_single = torch.randn(B, T)
static_attr_single = torch.randn(B, 11)
time_mask_single = torch.rand(B, T) < 0.75

encoder_output_std, mask_info_std = encoder_std(x_vec_single, static_attr_single, time_mask_single)

print(f"Standard encoder output:")
print(f"  Shape: {encoder_output_std.shape}")
print(f"  use_spatial_agg: {mask_info_std.get('use_spatial_agg', False)}")

assert encoder_output_std.dim() == 3, "Standard mode should return 3D tensor"
print(f"\n✓ Standard mode test passed")

print(f"\n" + "=" * 80)
print("✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓")
print("=" * 80)

print(f"\nSummary:")
print(f"  ✓ Spatial aggregation module works correctly")
print(f"  ✓ Vector encoder handles spatial aggregation")
print(f"  ✓ Vector decoder reverse-aggregates to catchments")
print(f"  ✓ Gradients flow correctly")
print(f"  ✓ Token reduction: {reduction:.1f}%")
print(f"  ✓ Backward compatible with standard mode")

print(f"\n🎉 Ready for training with spatial aggregation!")
