"""
测试优化后的数据加载性能

对比:
1. 原始模式(多个小h5文件)
2. 优化模式 - 无缓存
3. 优化模式 - 有缓存

Usage:
    python test_optimized_dataloader.py
"""

import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from datasets.multimodal_dataset import MultiModalHydroDataset
from datasets.data_utils import load_vector_data_from_parquet
from datasets.collate import MultiScaleMaskedCollate
from configs.mae_config import MAEConfig


def test_dataloader(
    dataset_name: str,
    dataset: MultiModalHydroDataset,
    num_batches: int = 50
):
    """
    Test dataloader performance

    Args:
        dataset_name: Name for display
        dataset: Dataset instance
        num_batches: Number of batches to test
    """
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")

    # Create collate function
    collate_fn = MultiScaleMaskedCollate(
        seq_len=30,
        mask_ratio=0.4,
        patch_size=10,
        land_mask_path='data/gb_temp_valid_mask_290x180.pt',
        land_threshold=0.5,
        mask_mode='unified',
        mode='train',
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Warmup
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break

    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    batch_times = []

    for i, batch in enumerate(dataloader):
        batch_start = time.time()

        # Simulate some processing
        _ = batch['precip'].shape

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if i >= num_batches - 1:
            break

    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average batch time: {avg_batch_time:.4f}s")
    print(f"  Throughput: {num_batches / total_time:.2f} batches/sec")
    print(f"  Min batch time: {min(batch_times):.4f}s")
    print(f"  Max batch time: {max(batch_times):.4f}s")

    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput': num_batches / total_time,
    }


def main():
    """Main test function"""

    print("\n" + "="*70)
    print("H5 Data Loading Performance Test")
    print("="*70)

    # Load config
    config = MAEConfig()

    # Load vector data
    print("\nLoading vector data...")
    vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
        config.vector_file,
        variables=['evaporation', 'discharge_vol'],
        start=datetime.strptime(config.train_start, '%Y-%m-%d'),
        end=datetime.strptime(config.train_end, '%Y-%m-%d'),
        nan_ratio=0.05,
    )

    evap_data = vector_data[:, :, 0].T
    riverflow_data = vector_data[:, :, 1].T

    print(f"✓ Vector data loaded: {evap_data.shape}")
    print(f"✓ Catchments: {len(catchment_ids)}")

    # Check if merged files exist
    merged_files_exist = (
        os.path.exists(config.precip_train_h5) and
        os.path.exists(config.soil_train_h5) and
        os.path.exists(config.temp_train_h5)
    )

    results = {}

    if not merged_files_exist:
        print("\n⚠️  Merged h5 files not found!")
        print(f"Expected locations:")
        print(f"  - {config.precip_train_h5}")
        print(f"  - {config.soil_train_h5}")
        print(f"  - {config.temp_train_h5}")
        print("\nTesting legacy mode only...")

        # Test legacy mode
        dataset_legacy = MultiModalHydroDataset(
            precip_dir=config.precip_dir,
            soil_dir=config.soil_dir,
            temp_dir=config.temp_dir,
            evap_data=evap_data,
            riverflow_data=riverflow_data,
            static_attr_file=config.static_attr_file,
            static_attr_vars=config.static_attrs,
            start_date=config.train_start,
            end_date=config.train_end,
            max_sequence_length=config.max_time_steps,
            stride=config.stride,
            catchment_ids=catchment_ids,
            stats_cache_path=config.stats_cache_path,
            land_mask_path=config.land_mask_path,
            split='train',
        )

        results['legacy'] = test_dataloader(
            "Legacy Mode (multiple h5 files)",
            dataset_legacy,
            num_batches=50
        )

    else:
        print("\n✓ Merged h5 files found! Testing all modes...\n")

        # Test 1: Legacy mode (for comparison)
        print("\n" + "="*70)
        print("TEST 1: Legacy Mode (baseline)")
        print("="*70)

        dataset_legacy = MultiModalHydroDataset(
            precip_dir=config.precip_dir,
            soil_dir=config.soil_dir,
            temp_dir=config.temp_dir,
            evap_data=evap_data,
            riverflow_data=riverflow_data,
            static_attr_file=config.static_attr_file,
            static_attr_vars=config.static_attrs,
            start_date=config.train_start,
            end_date=config.train_end,
            max_sequence_length=config.max_time_steps,
            stride=config.stride,
            catchment_ids=catchment_ids,
            stats_cache_path=config.stats_cache_path,
            land_mask_path=config.land_mask_path,
            split='train',
        )

        results['legacy'] = test_dataloader(
            "Legacy Mode (multiple h5 files)",
            dataset_legacy,
            num_batches=50
        )

        # Test 2: Optimized mode without memory cache
        print("\n" + "="*70)
        print("TEST 2: Optimized Mode (no memory cache)")
        print("="*70)

        dataset_optimized_no_cache = MultiModalHydroDataset(
            precip_h5=config.precip_train_h5,
            soil_h5=config.soil_train_h5,
            temp_h5=config.temp_train_h5,
            evap_data=evap_data,
            riverflow_data=riverflow_data,
            static_attr_file=config.static_attr_file,
            static_attr_vars=config.static_attrs,
            start_date=config.train_start,
            end_date=config.train_end,
            max_sequence_length=config.max_time_steps,
            stride=config.stride,
            catchment_ids=catchment_ids,
            stats_cache_path=config.stats_cache_path,
            land_mask_path=config.land_mask_path,
            split='train',
            cache_to_memory=False,
        )

        results['optimized_no_cache'] = test_dataloader(
            "Optimized Mode (on-demand loading)",
            dataset_optimized_no_cache,
            num_batches=50
        )

        # Test 3: Optimized mode with memory cache
        print("\n" + "="*70)
        print("TEST 3: Optimized Mode (memory cache)")
        print("="*70)

        dataset_optimized_cache = MultiModalHydroDataset(
            precip_h5=config.precip_train_h5,
            soil_h5=config.soil_train_h5,
            temp_h5=config.temp_train_h5,
            evap_data=evap_data,
            riverflow_data=riverflow_data,
            static_attr_file=config.static_attr_file,
            static_attr_vars=config.static_attrs,
            start_date=config.train_start,
            end_date=config.train_end,
            max_sequence_length=config.max_time_steps,
            stride=config.stride,
            catchment_ids=catchment_ids,
            stats_cache_path=config.stats_cache_path,
            land_mask_path=config.land_mask_path,
            split='train',
            cache_to_memory=True,
        )

        results['optimized_cache'] = test_dataloader(
            "Optimized Mode (memory cache)",
            dataset_optimized_cache,
            num_batches=50
        )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if 'legacy' in results:
        legacy_time = results['legacy']['avg_batch_time']
        print(f"\nLegacy Mode:")
        print(f"  Average batch time: {legacy_time:.4f}s")

        if 'optimized_no_cache' in results:
            speedup_no_cache = legacy_time / results['optimized_no_cache']['avg_batch_time']
            print(f"\nOptimized (no cache):")
            print(f"  Average batch time: {results['optimized_no_cache']['avg_batch_time']:.4f}s")
            print(f"  Speedup: {speedup_no_cache:.1f}x")

        if 'optimized_cache' in results:
            speedup_cache = legacy_time / results['optimized_cache']['avg_batch_time']
            print(f"\nOptimized (with cache):")
            print(f"  Average batch time: {results['optimized_cache']['avg_batch_time']:.4f}s")
            print(f"  Speedup: {speedup_cache:.1f}x")

    print("\n" + "="*70)
    print("✓ Test completed!")
    print("="*70)


if __name__ == '__main__':
    main()
