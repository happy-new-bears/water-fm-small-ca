"""
Test script for MultiModalMAE
"""

import sys
import os
sys.path.insert(0, '/Users/transformer/Desktop/water_code/water_fm')

import torch
from models.multimodal_mae import MultiModalMAE
from configs.mae_config import MAEConfig


def test_multimodal_mae():
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")

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
    print(f"\nRunning forward pass...")
    total_loss, loss_dict = model(batch)

    print(f"✓ Forward pass successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"    {key}: {value.item():.4f}")

    # Backward pass
    print(f"\nRunning backward pass...")
    total_loss.backward()

    print(f"✓ Backward pass successful")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())
    print(f"✓ Parameters with gradients: {has_grad}/{total_params_count}")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)


if __name__ == '__main__':
    test_multimodal_mae()
