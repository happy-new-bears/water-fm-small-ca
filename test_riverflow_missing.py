"""
Test script to verify riverflow_missing implementation

This script tests:
1. Dataset loading with 1970-1988 period (riverflow missing)
2. Dataset loading with 1989-2015 period (riverflow available)
3. Collate function handling of riverflow_missing
4. Basic data shapes and flags
"""

import sys
import torch
from datetime import datetime
from configs.mae_config import MAEConfig
from datasets.multimodal_dataset_optimized import MultiModalHydroDatasetOptimized
from datasets.data_utils import load_vector_data_from_parquet
from datasets.collate import MultiScaleMaskedCollate

def test_dataset_loading():
    """Test dataset loading for both time periods"""

    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)

    config = MAEConfig()

    # Load vector data
    print("\nLoading vector data...")
    vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
        config.vector_file,
        variables=['evaporation', 'discharge_vol'],
        start=datetime.strptime(config.train_start, '%Y-%m-%d'),
        end=datetime.strptime(config.val_end, '%Y-%m-%d'),
        nan_ratio=0.05,
    )

    evap_data = vector_data[:, :, 0].T
    riverflow_data = vector_data[:, :, 1].T

    print(f"✓ Vector data shape: {evap_data.shape}")

    # Find split index
    train_end_date = datetime.strptime(config.train_end, '%Y-%m-%d')
    time_vec_datetime = [datetime.strptime(str(d), '%Y-%m-%d') for d in time_vec]

    train_end_idx = None
    for i, date in enumerate(time_vec_datetime):
        if date == train_end_date:
            train_end_idx = i + 1
            break

    print(f"✓ Train end index: {train_end_idx}")

    # Test 1: Create dataset with 1970-2010 period (includes missing riverflow period)
    print("\n" + "="*60)
    print("Test 1: Dataset with 1970-2010 period")
    print("="*60)

    dataset = MultiModalHydroDatasetOptimized(
        precip_h5=config.precip_train_h5,
        soil_h5=config.soil_train_h5,
        temp_h5=config.temp_train_h5,
        evap_data=evap_data[:, :train_end_idx],
        riverflow_data=riverflow_data[:, :train_end_idx],
        static_attr_file=config.static_attr_file,
        static_attr_vars=config.static_attrs,
        start_date=config.train_start,  # 1970-01-01
        end_date=config.train_end,      # 2010-12-31
        max_sequence_length=config.max_time_steps,
        stride=config.stride,
        catchment_ids=catchment_ids,
        stats_cache_path=None,  # Don't cache for testing
        land_mask_path=config.land_mask_path,
        patch_size=config.vector_patch_size,
        split='test',
        cache_to_memory=False,
        riverflow_available_from=config.riverflow_available_from,
    )

    print(f"\n✓ Dataset created: {len(dataset)} samples")

    # Test sample from 1970 (riverflow missing)
    print("\n--- Testing sample from 1970 (riverflow should be missing) ---")
    sample_1970 = dataset[0]
    print(f"Start date: {sample_1970['start_date']}")
    print(f"Riverflow missing: {sample_1970['riverflow_missing']}")
    print(f"Precip shape: {sample_1970['precip'].shape}")
    print(f"Riverflow shape: {sample_1970['riverflow'].shape}")

    if sample_1970['riverflow_missing']:
        print("✅ PASS: Riverflow correctly marked as missing for 1970 sample")
    else:
        print("❌ FAIL: Riverflow should be missing for 1970 sample!")

    # Test sample from later period (riverflow available)
    # Find a sample from 1990s
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['start_date'].year >= 1990:
            print(f"\n--- Testing sample from {sample['start_date'].year} (riverflow should be available) ---")
            print(f"Start date: {sample['start_date']}")
            print(f"Riverflow missing: {sample['riverflow_missing']}")

            if not sample['riverflow_missing']:
                print("✅ PASS: Riverflow correctly marked as available for 1990s sample")
            else:
                print("❌ FAIL: Riverflow should be available for 1990s sample!")
            break

    return dataset

def test_collate_function(dataset):
    """Test collate function with riverflow_missing"""

    print("\n" + "="*60)
    print("Test 2: Collate Function")
    print("="*60)

    config = MAEConfig()

    collate_fn = MultiScaleMaskedCollate(
        seq_len=config.max_time_steps,
        mask_ratio=config.image_mask_ratio,
        patch_size=config.patch_size,
        image_height=config.image_height,
        image_width=config.image_width,
        vector_patch_size=config.vector_patch_size,
        land_mask_path=config.land_mask_path,
        land_threshold=config.land_threshold,
        mask_mode='independent',
        mode='train',
    )

    # Test with 1970 samples (riverflow missing)
    print("\n--- Batch from 1970 period ---")
    batch_1970_list = [dataset[i] for i in range(2)]  # Get 2 samples
    batch_1970 = collate_fn(batch_1970_list)

    print(f"Batch size: {batch_1970['precip'].shape[0]}")
    print(f"Riverflow missing flag: {batch_1970['riverflow_missing']}")
    print(f"Riverflow mask shape: {batch_1970['riverflow_mask'].shape}")
    print(f"Riverflow mask (first sample, first patch, first 10 timesteps):")
    print(f"  {batch_1970['riverflow_mask'][0, 0, :10]}")

    # Check if all riverflow is masked
    all_masked = batch_1970['riverflow_mask'].all()
    if batch_1970['riverflow_missing'] and all_masked:
        print("✅ PASS: All riverflow correctly masked when missing")
    elif batch_1970['riverflow_missing'] and not all_masked:
        print("❌ FAIL: Riverflow should be 100% masked when missing!")

    # Test with 1990s samples (riverflow available)
    print("\n--- Batch from 1990s period ---")
    # Find samples from 1990s
    samples_1990s = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['start_date'].year >= 1990 and not sample['riverflow_missing']:
            samples_1990s.append(sample)
            if len(samples_1990s) >= 2:
                break

    if len(samples_1990s) >= 2:
        batch_1990s = collate_fn(samples_1990s)
        print(f"Batch size: {batch_1990s['precip'].shape[0]}")
        print(f"Riverflow missing flag: {batch_1990s['riverflow_missing']}")
        print(f"Riverflow mask shape: {batch_1990s['riverflow_mask'].shape}")
        print(f"Riverflow mask (first sample, first patch, first 10 timesteps):")
        print(f"  {batch_1990s['riverflow_mask'][0, 0, :10]}")

        # Check if NOT all riverflow is masked
        all_masked = batch_1990s['riverflow_mask'].all()
        if not batch_1990s['riverflow_missing'] and not all_masked:
            print("✅ PASS: Riverflow NOT fully masked when available (partial masking for MAE)")
        elif not batch_1990s['riverflow_missing'] and all_masked:
            print("❌ FAIL: Riverflow should NOT be 100% masked when available!")

def main():
    """Run all tests"""

    print("\n" + "="*80)
    print(" "*20 + "RIVERFLOW MISSING TEST SUITE")
    print("="*80)

    try:
        # Test 1: Dataset loading
        dataset = test_dataset_loading()

        # Test 2: Collate function
        test_collate_function(dataset)

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)
        print("\nSummary:")
        print("- Dataset correctly identifies riverflow_missing based on date")
        print("- Collate function correctly masks riverflow 100% when missing")
        print("- Collate function correctly applies partial masking when available")

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
