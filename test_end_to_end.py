"""
End-to-End Test with Overfitting Verification (Small Scale for Local Machine)

This script performs:
1. End-to-end dataloader -> model -> loss -> backward test
2. Overfitting verification on a tiny dataset (should see loss drop significantly)
"""

import sys
import os
sys.path.insert(0, '/Users/transformer/Desktop/water_code/water_fm')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import numpy as np

from models.multimodal_mae import MultiModalMAE
from datasets.multimodal_dataset import MultiModalHydroDataset
from datasets.data_utils import load_vector_data_from_parquet
from datasets.collate import MultiScaleMaskedCollate
from configs.mae_config import MAEConfig


class TinyMAEConfig(MAEConfig):
    """Tiny configuration for local testing"""
    # Reduce model size
    d_model = 64
    decoder_dim = 32
    img_encoder_layers = 2
    vec_encoder_layers = 2
    decoder_layers = 1
    nhead = 4
    dropout = 0.0  # Disable dropout for overfitting

    # Small data
    max_time_steps = 10  # Only 10 days instead of 90
    batch_size = 2
    stride = 5  # Small stride

    # Training
    learning_rate = 1e-3  # Higher LR for faster overfitting
    epochs = 100  # We'll only run a few iterations


def create_tiny_dataset(config):
    """Create a very small dataset for testing"""

    print("=" * 60)
    print("Loading tiny dataset...")
    print("=" * 60)

    # Load vector data
    vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
        config.vector_file,
        variables=['evaporation', 'discharge_vol'],
        start=datetime.strptime('2000-01-01', '%Y-%m-%d'),
        end=datetime.strptime('2000-01-31', '%Y-%m-%d'),  # Only 1 month!
        nan_ratio=0.05,
    )

    evap_data = vector_data[:, :, 0].T
    riverflow_data = vector_data[:, :, 1].T

    # Only use first 2 catchments
    num_catchments = min(2, len(catchment_ids))
    evap_data = evap_data[:num_catchments]
    riverflow_data = riverflow_data[:num_catchments]
    catchment_ids = catchment_ids[:num_catchments]

    print(f"✓ Using {num_catchments} catchments")
    print(f"✓ Time range: {len(time_vec)} days")

    # Create dataset
    dataset = MultiModalHydroDataset(
        precip_dir=config.precip_dir,
        soil_dir=config.soil_dir,
        temp_dir=config.temp_dir,
        evap_data=evap_data,
        riverflow_data=riverflow_data,
        static_attr_file=config.static_attr_file,
        static_attr_vars=config.static_attrs,
        start_date='2000-01-01',
        end_date='2000-01-31',
        max_sequence_length=config.max_time_steps,
        stride=config.stride,
        catchment_ids=catchment_ids,
        stats_cache_path=config.stats_cache_path,
        land_mask_path=config.land_mask_path,
        split='train',
    )

    print(f"✓ Dataset created: {len(dataset)} samples")

    # Only use first 10 samples for overfitting test
    tiny_dataset = Subset(dataset, range(min(10, len(dataset))))
    print(f"✓ Using tiny subset: {len(tiny_dataset)} samples for overfitting")

    return tiny_dataset


def test_dataloader(config):
    """Test that dataloader works correctly"""

    print("\n" + "=" * 60)
    print("TEST 1: DataLoader Functionality")
    print("=" * 60)

    tiny_dataset = create_tiny_dataset(config)

    # Create collate function
    collate_fn = MultiScaleMaskedCollate(
        seq_len=config.max_time_steps,
        mask_ratio=config.image_mask_ratio,
        patch_size=config.patch_size,
        land_mask_path=config.land_mask_path,
        land_threshold=config.land_threshold,
        mask_mode='unified',
        mode='train',
    )

    # Create dataloader
    dataloader = DataLoader(
        tiny_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    print(f"✓ DataLoader created")

    # Load one batch
    batch = next(iter(dataloader))

    print(f"✓ Batch loaded successfully")
    print(f"  Batch size: {batch['precip'].shape[0]}")
    print(f"  Sequence length: {batch['seq_len']}")
    print(f"  Image shape: {batch['precip'].shape}")
    print(f"  Vector shape: {batch['evap'].shape}")
    print(f"  Static attr shape: {batch['static_attr'].shape}")

    # Check masks
    print(f"\n✓ Mask statistics:")
    print(f"  Image mask ratio: {batch['precip_mask'].float().mean():.2%}")
    print(f"  Vector mask ratio: {batch['evap_mask'].float().mean():.2%}")

    return dataloader


def test_model_forward_backward(config, dataloader):
    """Test model forward and backward pass"""

    print("\n" + "=" * 60)
    print("TEST 2: Model Forward & Backward")
    print("=" * 60)

    # Get valid patch indices
    land_mask = torch.load(config.land_mask_path)
    patch_size = config.patch_size
    num_patches_h = config.image_height // patch_size
    num_patches_w = config.image_width // patch_size

    valid_patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = land_mask[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            land_ratio = patch.sum().item() / (patch_size * patch_size)
            if land_ratio >= config.land_threshold:
                valid_patches.append(i * num_patches_w + j)

    valid_patch_indices = torch.tensor(valid_patches, dtype=torch.long)

    # Create model
    model = MultiModalMAE(config, valid_patch_indices)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")

    # Get a batch
    batch = next(iter(dataloader))

    # Forward pass
    print(f"\n✓ Running forward pass...")
    total_loss, loss_dict = model(batch)

    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"    {key}: {value.item():.4f}")

    # Backward pass
    print(f"\n✓ Running backward pass...")
    total_loss.backward()

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params_count}")

    assert has_grad == total_params_count, "Not all parameters have gradients!"

    print(f"\n✓ Forward & backward pass successful!")

    return model


def test_overfitting(config, dataloader, model):
    """Test that model can overfit on tiny dataset"""

    print("\n" + "=" * 60)
    print("TEST 3: Overfitting Verification")
    print("=" * 60)
    print("Training on tiny dataset to verify model can learn...")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # Get a single batch for overfitting
    batch = next(iter(dataloader))
    print(f"✓ Using single batch with {batch['precip'].shape[0]} samples")

    # Record initial loss
    model.eval()
    with torch.no_grad():
        initial_loss, _ = model(batch)
        initial_loss_value = initial_loss.item()

    print(f"\n✓ Initial loss: {initial_loss_value:.4f}")

    # Training loop
    model.train()
    num_steps = 100
    log_interval = 10

    losses = []

    print(f"\nTraining for {num_steps} steps...")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward
        total_loss, loss_dict = model(batch)

        # Backward
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        # Log
        if (step + 1) % log_interval == 0:
            print(f"  Step {step+1}/{num_steps}: Loss = {total_loss.item():.4f}")

    # Final loss
    final_loss_value = losses[-1]

    print(f"\n" + "=" * 60)
    print("Overfitting Results:")
    print("=" * 60)
    print(f"  Initial loss: {initial_loss_value:.4f}")
    print(f"  Final loss:   {final_loss_value:.4f}")
    print(f"  Reduction:    {(1 - final_loss_value/initial_loss_value)*100:.1f}%")

    # Check if loss decreased significantly
    reduction_ratio = final_loss_value / initial_loss_value

    if reduction_ratio < 0.5:
        print(f"\n✓✓✓ SUCCESS: Loss reduced to {reduction_ratio*100:.1f}% of initial value!")
        print(f"    Model can successfully learn and overfit on data.")
        return True
    else:
        print(f"\n✗✗✗ WARNING: Loss only reduced to {reduction_ratio*100:.1f}%")
        print(f"    Expected <50% for successful overfitting.")
        return False


def main():
    """Run all end-to-end tests"""

    print("\n" + "=" * 80)
    print("END-TO-END TEST WITH OVERFITTING VERIFICATION")
    print("(Small Scale for Local Machine)")
    print("=" * 80)

    # Use tiny config
    config = TinyMAEConfig()

    print(f"\nTest Configuration:")
    print(f"  Model size: d_model={config.d_model}, decoder_dim={config.decoder_dim}")
    print(f"  Encoder layers: img={config.img_encoder_layers}, vec={config.vec_encoder_layers}")
    print(f"  Decoder layers: {config.decoder_layers}")
    print(f"  Sequence length: {config.max_time_steps} days")
    print(f"  Batch size: {config.batch_size}")

    # Test 1: DataLoader
    dataloader = test_dataloader(config)

    # Test 2: Model forward & backward
    model = test_model_forward_backward(config, dataloader)

    # Test 3: Overfitting
    success = test_overfitting(config, dataloader, model)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ Test 1: DataLoader functionality - PASSED")
    print("✓ Test 2: Model forward & backward - PASSED")
    if success:
        print("✓ Test 3: Overfitting verification - PASSED")
    else:
        print("⚠ Test 3: Overfitting verification - NEEDS ATTENTION")

    print("\n" + "=" * 80)
    if success:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Model is ready for full-scale training!")
    else:
        print("⚠⚠⚠ OVERFITTING TEST DID NOT PASS ⚠⚠⚠")
        print("Please check model implementation or increase training steps.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
