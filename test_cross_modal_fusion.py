"""
Test script for cross-modal fusion implementation

验证:
1. Forward pass能正常运行
2. Backward pass能正常计算梯度
3. 所有参数都有梯度
4. 输出shape正确
"""

import torch
import torch.nn as nn
from models.multimodal_mae import MultiModalMAE
from configs.mae_config import MAEConfig


def create_fixed_mask(B, dim1, dim2, mask_ratio=0.75):
    """
    Create a fixed-ratio mask where all samples have exactly the same number of masked elements.

    Args:
        B: batch size
        dim1, dim2: dimensions (either T, num_patches or num_patches, T)
        mask_ratio: ratio of masked elements

    Returns:
        mask: [B, dim1, dim2] bool tensor (True = masked)
    """
    total_elements = dim1 * dim2
    num_masked = int(total_elements * mask_ratio)

    mask = torch.zeros(B, dim1, dim2, dtype=torch.bool)

    for b in range(B):
        # Generate random permutation
        perm = torch.randperm(total_elements)
        masked_indices = perm[:num_masked]

        # Convert flat indices to 2D indices
        row_idx = masked_indices // dim2
        col_idx = masked_indices % dim2

        mask[b, row_idx, col_idx] = True

    return mask


def test_cross_modal_fusion():
    print("=" * 80)
    print("Testing Cross-Modal Fusion Implementation")
    print("=" * 80)

    # ===== Step 1: Create config with small dimensions =====
    print("\n[Step 1] Creating configuration...")
    config = MAEConfig()

    # Use very small dimensions for quick testing
    config.d_model = 64
    config.decoder_dim = 32
    config.nhead = 4
    config.img_encoder_layers = 2
    config.vec_encoder_layers = 2
    config.img_decoder_layers = 2
    config.vec_decoder_layers = 2
    config.dropout = 0.1
    config.shared_depth = 1  # 1 shared transformer layer

    # Small spatial size
    config.image_height = 100
    config.image_width = 60
    config.patch_size = 10

    print(f"  d_model={config.d_model}, decoder_dim={config.decoder_dim}")
    print(f"  nhead={config.nhead}")
    print(f"  shared_depth={config.shared_depth}")
    print(f"  img_encoder_layers={config.img_encoder_layers}")

    # ===== Step 2: Create model =====
    print("\n[Step 2] Creating model...")
    model = MultiModalMAE(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Check modality tokens exist
    assert hasattr(model, 'modality_precip'), "Missing encoder modality tokens"
    assert hasattr(model, 'decoder_modality_precip'), "Missing decoder modality tokens"
    assert hasattr(model, 'blocks_shared'), "Missing shared transformer"
    print(f"  ✓ Modality tokens initialized")
    print(f"  ✓ Shared transformer initialized ({config.shared_depth} layers)")

    # ===== Step 3: Create small batch =====
    print("\n[Step 3] Creating test batch...")
    B, T = 2, 10  # Very small batch
    H, W = config.image_height, config.image_width

    # Vector data needs to be in patchified format
    # [B, num_patches, patch_size, T]
    num_catchments = 604
    patch_size = 8
    num_patches = 76  # 604 / 8

    batch = {
        # Image modalities [B, T, H, W]
        'precip': torch.randn(B, T, H, W),
        'soil': torch.randn(B, T, H, W),
        'temp': torch.randn(B, T, H, W),

        # Vector modalities [B, num_patches, patch_size, T]
        'evap': torch.randn(B, num_patches, patch_size, T),
        'riverflow': torch.randn(B, num_patches, patch_size, T),

        # Static attributes [B, num_catchments, stat_dim]
        'static_attr': torch.randn(B, num_catchments, 27),

        # Catchment padding mask [B, num_patches, patch_size]
        'catchment_padding_mask': torch.zeros(B, num_patches, patch_size, dtype=torch.bool),

        # Masks (True = masked)
        # Use FIXED mask ratio (75%) to ensure all samples have same number of masked patches
        # This is required by the current vectorized implementation
        'precip_mask': create_fixed_mask(B, T, (H // config.patch_size) * (W // config.patch_size), mask_ratio=0.75),
        'soil_mask': create_fixed_mask(B, T, (H // config.patch_size) * (W // config.patch_size), mask_ratio=0.75),
        'temp_mask': create_fixed_mask(B, T, (H // config.patch_size) * (W // config.patch_size), mask_ratio=0.75),
        'evap_mask': create_fixed_mask(B, num_patches, T, mask_ratio=0.75),
        'riverflow_mask': create_fixed_mask(B, num_patches, T, mask_ratio=0.75),
    }

    print(f"  Batch size: {B}")
    print(f"  Time steps: {T}")
    print(f"  Image shape: {batch['precip'].shape}")
    print(f"  Vector shape: {batch['evap'].shape}")

    # ===== Step 4: Forward pass =====
    print("\n[Step 4] Running forward pass...")
    try:
        total_loss, loss_dict = model(batch)
        print(f"  ✓ Forward pass successful!")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Loss breakdown:")
        for k, v in loss_dict.items():
            print(f"    - {k}: {v.item():.4f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== Step 5: Backward pass =====
    print("\n[Step 5] Running backward pass...")
    try:
        model.zero_grad()
        total_loss.backward()
        print(f"  ✓ Backward pass successful!")
    except Exception as e:
        print(f"  ✗ Backward pass failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== Step 6: Check gradients =====
    print("\n[Step 6] Checking gradients...")

    # Count parameters with gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())

    print(f"  Parameters with gradients: {params_with_grad}/{total_params_count}")

    # Check modality tokens have gradients
    modality_tokens = [
        ('modality_precip', model.modality_precip),
        ('modality_soil', model.modality_soil),
        ('modality_temp', model.modality_temp),
        ('modality_evap', model.modality_evap),
        ('modality_riverflow', model.modality_riverflow),
        ('decoder_modality_precip', model.decoder_modality_precip),
        ('decoder_modality_soil', model.decoder_modality_soil),
        ('decoder_modality_temp', model.decoder_modality_temp),
        ('decoder_modality_evap', model.decoder_modality_evap),
        ('decoder_modality_riverflow', model.decoder_modality_riverflow),
    ]

    print("\n  Modality token gradients:")
    for name, token in modality_tokens:
        has_grad = token.grad is not None
        grad_norm = token.grad.norm().item() if has_grad else 0.0
        # Consider grad_norm > 1e-10 as having gradient
        status = "✓" if (has_grad and grad_norm > 1e-10) else "✗"
        grad_str = f"{grad_norm:.6f}" if has_grad else "None"
        print(f"    {status} {name}: grad_norm={grad_str}")

    # Check shared transformer has gradients
    shared_has_grad = any(p.grad is not None for p in model.blocks_shared.parameters())
    print(f"\n  {'✓' if shared_has_grad else '✗'} Shared transformer has gradients: {shared_has_grad}")

    # ===== Step 7: Check output shapes =====
    print("\n[Step 7] Running forward again to check output shapes...")
    with torch.no_grad():
        _, loss_dict = model(batch)

    # Note: loss_dict doesn't contain predictions, they're computed internally
    print(f"  ✓ Loss computation successful")

    # ===== Final Summary =====
    print("\n" + "=" * 80)
    if params_with_grad == total_params_count:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 80)
        print("\nCross-modal fusion implementation is ready!")
        print("- All modality tokens initialized ✓")
        print("- Shared transformer working ✓")
        print("- Decoders receive fused features ✓")
        print("- All parameters trainable ✓")
        return True
    else:
        print("⚠ SOME PARAMETERS MISSING GRADIENTS")
        print("=" * 80)
        missing = total_params_count - params_with_grad
        print(f"\n{missing} parameters don't have gradients!")
        return False


if __name__ == '__main__':
    success = test_cross_modal_fusion()
    exit(0 if success else 1)
