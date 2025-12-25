"""
Test script for Multi-modal Hydrology DataLoader

This script:
1. Loads vector data from parquet
2. Creates datasets with normalization
3. Tests DataLoader with different configurations
4. Validates shapes, masks, and data integrity
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from datasets.data_utils import load_vector_data_from_parquet
from datasets.multimodal_dataset import MultiModalHydroDataset
from datasets.collate import MultiScaleMaskedCollate


# Configuration
DATA_CONFIG = {
    'precip_dir': '/Users/transformer/Desktop/water_data/new_version/precipitation_processed',
    'soil_dir': '/Users/transformer/Desktop/water_data/new_version/soil_moisture_processed',
    'temp_dir': '/Users/transformer/Desktop/water_data/new_version/temperature_processed',
    'vector_file': '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet',
    'static_attr_file': '/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/Catchment_attributes_nrfa.csv',
    'land_mask_path': '/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt',
}

STATIC_ATTRS = [
    "latitude", "longitude",
    "minimum-altitude", "maximum-altitude", "50-percentile-altitude",
    "10-percentile-altitude", "90-percentile-altitude",
    "catchment-area", "dpsbar",
    "propwet", "bfihost",
]

# Time periods
TRAIN_START = '1989-01-01'
TRAIN_END = '2010-12-31'
VAL_START = '2011-01-01'
VAL_END = '2015-12-31'


def test_vector_data_loading():
    """Test loading vector data from parquet"""
    print("\n" + "="*80)
    print("TEST 1: Loading Vector Data from Parquet")
    print("="*80)

    try:
        # Load vector data
        vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
            DATA_CONFIG['vector_file'],
            variables=['evaporation', 'discharge_vol'],  # Use correct column name
            start=datetime.strptime(TRAIN_START, '%Y-%m-%d'),
            end=datetime.strptime(TRAIN_END, '%Y-%m-%d'),
            nan_ratio=0.05,
        )

        print(f"✓ Vector data shape: {vector_data.shape}")
        print(f"✓ Time vector shape: {time_vec.shape}")
        print(f"✓ Catchment IDs shape: {catchment_ids.shape}")
        print(f"✓ Variable names: {var_names}")
        print(f"✓ Date range: {time_vec[0]} to {time_vec[-1]}")

        # Extract individual modalities
        evap_data = vector_data[:, :, 0].T  # [num_catchments, num_days]
        riverflow_data = vector_data[:, :, 1].T  # [num_catchments, num_days]

        print(f"✓ Evaporation data shape: {evap_data.shape}")
        print(f"✓ Riverflow data shape: {riverflow_data.shape}")

        # Check for NaN
        print(f"✓ Evaporation NaN count: {np.isnan(evap_data).sum()}")
        print(f"✓ Riverflow NaN count: {np.isnan(riverflow_data).sum()}")

        return evap_data, riverflow_data, catchment_ids, True

    except Exception as e:
        print(f"✗ Error loading vector data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False


def test_dataset_creation(evap_data, riverflow_data, catchment_ids):
    """Test dataset creation with normalization"""
    print("\n" + "="*80)
    print("TEST 2: Creating Dataset with Normalization")
    print("="*80)

    try:
        # Create dataset
        dataset = MultiModalHydroDataset(
            # Image modalities
            precip_dir=DATA_CONFIG['precip_dir'],
            soil_dir=DATA_CONFIG['soil_dir'],
            temp_dir=DATA_CONFIG['temp_dir'],
            # Vector modalities
            evap_data=evap_data,
            riverflow_data=riverflow_data,
            # Static attributes
            static_attr_file=DATA_CONFIG['static_attr_file'],
            static_attr_vars=STATIC_ATTRS,
            # Time range
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            # Parameters
            max_sequence_length=90,
            stride=30,  # Stride for sliding window (default: 30 days)
            catchment_ids=catchment_ids,
            # Normalization
            stats_cache_path='cache/normalization_stats.pt',
            land_mask_path=DATA_CONFIG['land_mask_path'],
            split='train',
        )

        print(f"✓ Dataset created successfully")
        print(f"✓ Number of valid samples: {len(dataset)}")
        print(f"✓ Number of catchments: {dataset.num_catchments}")
        print(f"✓ Number of days: {dataset.num_days}")

        return dataset, True

    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_single_sample(dataset):
    """Test loading a single sample"""
    print("\n" + "="*80)
    print("TEST 3: Loading Single Sample")
    print("="*80)

    try:
        # Get first sample
        sample = dataset[0]

        print(f"✓ Sample keys: {list(sample.keys())}")

        # Check shapes
        print(f"✓ Precip shape: {sample['precip'].shape}")
        print(f"✓ Soil shape: {sample['soil'].shape}")
        print(f"✓ Temp shape: {sample['temp'].shape}")
        print(f"✓ Evap shape: {sample['evap'].shape}")
        print(f"✓ Riverflow shape: {sample['riverflow'].shape}")
        print(f"✓ Static attr shape: {sample['static_attr'].shape}")

        # Check data ranges (should be normalized)
        print(f"✓ Precip range: [{sample['precip'].min():.2f}, {sample['precip'].max():.2f}]")
        print(f"✓ Evap range: [{sample['evap'].min():.2f}, {sample['evap'].max():.2f}]")
        print(f"✓ Riverflow range: [{sample['riverflow'].min():.2f}, {sample['riverflow'].max():.2f}]")

        # Check for NaN
        has_nan = any([
            np.isnan(sample['precip']).any(),
            np.isnan(sample['soil']).any(),
            np.isnan(sample['temp']).any(),
            np.isnan(sample['evap']).any(),
            np.isnan(sample['riverflow']).any(),
        ])

        if has_nan:
            print(f"✗ Warning: Found NaN values in sample")
        else:
            print(f"✓ No NaN values in sample")

        return True

    except Exception as e:
        print(f"✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(dataset):
    """Test DataLoader with MAE-style masking"""
    print("\n" + "="*80)
    print("TEST 4: Testing DataLoader with MAE-style Masking")
    print("="*80)

    try:
        # Create collate function with fixed length and mask ratio
        collate_fn = MultiScaleMaskedCollate(
            seq_len=90,           # Fixed sequence length
            mask_ratio=0.75,      # Mask 75% (like MAE paper)
            patch_size=10,
            land_mask_path=DATA_CONFIG['land_mask_path'],  # Use land mask
            land_threshold=0.5,   # Patches with >=50% land
            mask_mode='unified',  # Unified mask across modalities
            mode='train',
        )

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for debugging
            drop_last=True,
        )

        print(f"✓ DataLoader created successfully")

        # Test loading a batch
        batch = next(iter(loader))

        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Sequence length: {batch['seq_len']}")

        # Check shapes
        B = len(batch['catchment_ids'])
        T = batch['seq_len']
        print(f"✓ Batch size: {B}")

        print(f"✓ Precip shape: {batch['precip'].shape}")
        print(f"✓ Soil shape: {batch['soil'].shape}")
        print(f"✓ Temp shape: {batch['temp'].shape}")
        print(f"✓ Evap shape: {batch['evap'].shape}")
        print(f"✓ Riverflow shape: {batch['riverflow'].shape}")
        print(f"✓ Static attr shape: {batch['static_attr'].shape}")

        # Check mask shapes - KEY DIFFERENCE!
        print(f"✓ Precip mask shape: {batch['precip_mask'].shape}")  # [B, T, 522] for images
        print(f"✓ Evap mask shape: {batch['evap_mask'].shape}")      # [B, T] for vectors

        # Calculate mask ratios
        precip_mask_ratio = batch['precip_mask'].float().mean().item()
        evap_mask_ratio = batch['evap_mask'].float().mean().item()
        print(f"✓ Image (precip) mask ratio: {precip_mask_ratio:.3f}")
        print(f"✓ Vector (evap) mask ratio: {evap_mask_ratio:.3f}")

        # Check unified mask (precip, soil, temp should have same mask)
        same_mask = (batch['precip_mask'] == batch['soil_mask']).all()
        print(f"✓ Precip and Soil masks identical: {same_mask.item()}")

        # Test multiple batches
        print("\n✓ Testing multiple batches:")
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            mask_ratio = batch['precip_mask'].float().mean().item()
            print(f"  Batch {i+1}: seq_len={batch['seq_len']}, "
                  f"image_mask_ratio={mask_ratio:.3f}")

        return True

    except Exception as e:
        print(f"✗ Error in DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_mode(dataset):
    """Test validation mode (no masking)"""
    print("\n" + "="*80)
    print("TEST 5: Testing Validation Mode (No Masking)")
    print("="*80)

    try:
        # Create validation collate function
        val_collate = MultiScaleMaskedCollate(
            seq_len=90,          # Fixed length
            mask_ratio=0.0,      # Not used in val mode
            land_mask_path=DATA_CONFIG['land_mask_path'],
            land_threshold=0.5,
            mode='val',          # Validation mode
        )

        # Create DataLoader
        val_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=0,
        )

        # Test batch
        batch = next(iter(val_loader))

        print(f"✓ Validation batch loaded")
        print(f"✓ Sequence length: {batch['seq_len']}")

        # Check image mask
        img_mask_ratio = batch['precip_mask'].float().mean().item()
        vec_mask_ratio = batch['evap_mask'].float().mean().item()

        print(f"✓ Image mask ratio: {img_mask_ratio:.3f} (should be 0.0)")
        print(f"✓ Vector mask ratio: {vec_mask_ratio:.3f} (should be 0.0)")

        if img_mask_ratio == 0.0 and vec_mask_ratio == 0.0:
            print(f"✓ Masks correctly disabled in validation mode")
        else:
            print(f"✗ Warning: Masks not disabled in validation mode")

        return True

    except Exception as e:
        print(f"✗ Error in validation mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MULTI-MODAL HYDROLOGY DATALOADER TESTS")
    print("="*80)

    results = []

    # Test 1: Load vector data
    evap_data, riverflow_data, catchment_ids, success = test_vector_data_loading()
    results.append(("Vector Data Loading", success))

    if not success:
        print("\n✗ Stopping tests due to vector data loading failure")
        return

    # Test 2: Create dataset
    dataset, success = test_dataset_creation(evap_data, riverflow_data, catchment_ids)
    results.append(("Dataset Creation", success))

    if not success:
        print("\n✗ Stopping tests due to dataset creation failure")
        return

    # Test 3: Single sample
    success = test_single_sample(dataset)
    results.append(("Single Sample Loading", success))

    # Test 4: DataLoader
    success = test_dataloader(dataset)
    results.append(("DataLoader with Masking", success))

    # Test 5: Validation mode
    success = test_validation_mode(dataset)
    results.append(("Validation Mode", success))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)

    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
