"""
Quick test script to verify training works without spatial aggregation
Uses legacy mode (directory-based h5 files) for testing
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.mae_config import MAEConfig
from datasets.multimodal_dataset import MultiModalHydroDataset
from datasets.collate import MultiScaleMaskedCollate
from models.multimodal_mae import MultiModalMAE

print("="*80)
print("QUICK TRAINING TEST (WITHOUT SPATIAL AGGREGATION)")
print("="*80)

# Load config
config = MAEConfig()

# Verify use_spatial_agg is False
print(f"\n✓ Config check: use_spatial_agg = {hasattr(config, 'use_spatial_agg') and config.use_spatial_agg}")
if hasattr(config, 'use_spatial_agg'):
    print("  ERROR: use_spatial_agg still exists in config!")
    sys.exit(1)
else:
    print("  ✓ use_spatial_agg has been removed from config")

# Load vector data
print(f"\nLoading vector data from: {config.vector_file}")
if not os.path.exists(config.vector_file):
    print(f"ERROR: Vector file not found: {config.vector_file}")
    sys.exit(1)

df = pd.read_parquet(config.vector_file)
print(f"  ✓ Loaded: {df.shape}")

# Extract data
# Pivot to get [num_catchments, num_days] format
df_pivot = df.pivot(index='ID', columns='date', values=['evaporation', 'discharge_vol'])
evap_data = df_pivot['evaporation'].values  # [num_catchments, num_days]
riverflow_data = df_pivot['discharge_vol'].values  # [num_catchments, num_days]
catchment_ids = df_pivot.index.values

print(f"  Evaporation shape: {evap_data.shape}")
print(f"  Riverflow shape: {riverflow_data.shape}")
print(f"  Catchments: {len(catchment_ids)}")

# Create small subset for quick testing (use only first 5 catchments)
num_test_catchments = 5
evap_data_small = evap_data[:num_test_catchments]
riverflow_data_small = riverflow_data[:num_test_catchments]
catchment_ids_small = catchment_ids[:num_test_catchments]

print(f"\n✓ Using {num_test_catchments} catchments for quick test")

# Create dataset (LEGACY MODE - no merged h5)
print(f"\nCreating dataset (legacy mode)...")
print(f"  Start date: {config.train_start}")
print(f"  End date: {config.train_end}")

try:
    dataset = MultiModalHydroDataset(
        # Use directory mode (legacy)
        precip_dir=config.precip_dir,
        soil_dir=config.soil_dir,
        temp_dir=config.temp_dir,
        # Vector data
        evap_data=evap_data_small,
        riverflow_data=riverflow_data_small,
        # Static attributes
        static_attr_file=config.static_attr_file,
        static_attr_vars=config.static_attrs,
        # Time range
        start_date=config.train_start,
        end_date=config.train_end,
        # Sampling parameters
        max_sequence_length=config.max_time_steps,
        stride=config.stride,
        # Catchment configuration
        catchment_ids=catchment_ids_small,
        # Normalization
        stats_cache_path=None,  # Don't use cache for testing
        land_mask_path=config.land_mask_path,
        # Other
        split='train',
        cache_to_memory=False,  # Use legacy mode
    )
    print(f"  ✓ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"  ERROR creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create collate function
print(f"\nCreating collate function...")
collate_fn = MultiScaleMaskedCollate(
    seq_len=config.max_time_steps,
    mask_ratio=0.75,
    patch_size=config.patch_size,
    image_height=config.image_height,
    image_width=config.image_width,
    land_mask_path=config.land_mask_path,
    mask_mode='unified',
    mode='train',
)
print(f"  ✓ Collate function created")

# Create dataloader (small batch for testing)
batch_size = 2
print(f"\nCreating dataloader (batch_size={batch_size})...")
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # No multiprocessing for testing
    collate_fn=collate_fn,
    pin_memory=False,
)
print(f"  ✓ Dataloader created")

# Create model
print(f"\nCreating model...")
try:
    # Load land mask
    land_mask = torch.load(config.land_mask_path)

    # Get valid patch indices
    num_patches = (config.image_height // config.patch_size) * \
                 (config.image_width // config.patch_size)
    valid_patches = []
    patch_size = config.patch_size

    for i in range(config.image_height // patch_size):
        for j in range(config.image_width // patch_size):
            patch = land_mask[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            land_ratio = patch.sum() / (patch_size * patch_size)
            if land_ratio >= 0.5:
                valid_patches.append(i * (config.image_width // patch_size) + j)

    valid_patch_indices = torch.tensor(valid_patches, dtype=torch.long)
    print(f"  Valid patches: {len(valid_patches)}/{num_patches}")

    model = MultiModalMAE(config, valid_patch_indices)
    print(f"  ✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"  ERROR creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
print(f"\nTesting forward pass...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = model.to(device)
model.train()

try:
    # Get one batch
    print(f"  Loading batch...")
    batch = next(iter(dataloader))

    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    print(f"  Batch shapes:")
    for key in ['precip', 'soil', 'temp', 'evap', 'riverflow', 'static_attr']:
        print(f"    {key}: {batch[key].shape}")

    # Forward pass
    print(f"  Running forward pass...")
    total_loss, loss_dict = model(batch)

    print(f"\n✓ Forward pass successful!")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.4f}")

    # Test backward pass
    print(f"\n  Testing backward pass...")
    total_loss.backward()
    print(f"  ✓ Backward pass successful!")

except Exception as e:
    print(f"\n  ERROR in forward/backward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("="*80)
print("\nThe model is working correctly without spatial aggregation!")
print("You can now run the full training with: python train_mae.py")
