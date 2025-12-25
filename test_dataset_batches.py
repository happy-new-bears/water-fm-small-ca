"""
Test script to verify dataset batch outputs
Prints detailed information about the first two batches
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultiModalHydroDataset
from datasets.collate import MultiScaleMaskedCollate

def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_batch_info(batch_idx, batch):
    """Print detailed information about a batch"""
    print_separator(f"BATCH {batch_idx}")

    # Print time range
    print(f"\nTime Information:")
    print(f"  Start dates: {batch['start_date']}")

    # Print image modality shapes
    print(f"\nImage Modalities (shape):")
    print(f"  precip:      {batch['precip'].shape}")
    print(f"  soil:        {batch['soil'].shape}")
    print(f"  temp:        {batch['temp'].shape}")

    # Print vector modality shapes
    print(f"\nVector Modalities (patchified):")
    print(f"  evap:        {batch['evap'].shape}")
    print(f"  riverflow:   {batch['riverflow'].shape}")
    print(f"  static_attr: {batch['static_attr'].shape}")

    # Print mask shapes
    print(f"\nMasks:")
    print(f"  precip_mask:     {batch['precip_mask'].shape}")
    print(f"  soil_mask:       {batch['soil_mask'].shape}")
    print(f"  temp_mask:       {batch['temp_mask'].shape}")
    print(f"  evap_mask:       {batch['evap_mask'].shape}")
    print(f"  riverflow_mask:  {batch['riverflow_mask'].shape}")
    print(f"  catchment_padding_mask: {batch['catchment_padding_mask'].shape}")

    # Print patch info
    print(f"\nPatch Information:")
    print(f"  num_patches: {batch['num_patches']}")
    print(f"  patch_size:  {batch['patch_size']}")

    # Print mask statistics
    print(f"\nMask Statistics (True = masked):")
    print(f"  precip_mask:     {batch['precip_mask'].sum().item()} / {batch['precip_mask'].numel()} masked ({100 * batch['precip_mask'].float().mean():.1f}%)")
    print(f"  soil_mask:       {batch['soil_mask'].sum().item()} / {batch['soil_mask'].numel()} masked ({100 * batch['soil_mask'].float().mean():.1f}%)")
    print(f"  temp_mask:       {batch['temp_mask'].sum().item()} / {batch['temp_mask'].numel()} masked ({100 * batch['temp_mask'].float().mean():.1f}%)")
    print(f"  evap_mask:       {batch['evap_mask'].sum().item()} / {batch['evap_mask'].numel()} masked ({100 * batch['evap_mask'].float().mean():.1f}%)")
    print(f"  riverflow_mask:  {batch['riverflow_mask'].sum().item()} / {batch['riverflow_mask'].numel()} masked ({100 * batch['riverflow_mask'].float().mean():.1f}%)")

    # Print data range
    print(f"\nData Value Ranges (normalized):")
    print(f"  precip:    [{batch['precip'].min():.3f}, {batch['precip'].max():.3f}]")
    print(f"  soil:      [{batch['soil'].min():.3f}, {batch['soil'].max():.3f}]")
    print(f"  temp:      [{batch['temp'].min():.3f}, {batch['temp'].max():.3f}]")
    print(f"  evap:      [{batch['evap'].min():.3f}, {batch['evap'].max():.3f}]")
    print(f"  riverflow: [{batch['riverflow'].min():.3f}, {batch['riverflow'].max():.3f}]")

    print()


def main():
    print_separator("Dataset Batch Verification Test")

    # Configuration (hardcoded)
    batch_size = 16
    max_sequence_length = 30
    stride = 30
    mask_ratio = 0.4
    start_date = '1989-01-01'
    end_date = '2010-12-31'

    # File paths
    precip_h5 = '/Users/transformer/Desktop/water_data/new_version/precipitation_train_1989_2010.h5'
    soil_h5 = '/Users/transformer/Desktop/water_data/new_version/soil_moisture_train_1989_2010.h5'
    temp_h5 = '/Users/transformer/Desktop/water_data/new_version/temperature_train_1989_2010.h5'
    static_attr_file = '/Users/transformer/Desktop/water_data/Catchment_attributes/Catchment_attributes_nrfa.csv'
    stats_cache_path = '/Users/transformer/Desktop/water_data/cache/stats_cache.pt'
    land_mask_path = '/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt'

    static_attr_vars = [
        'catchment-area', 'dpsbar', 'propwet', 'bfihost',
        'farl', 'sprhost', 'altbar'
    ]

    print("\nConfiguration:")
    print(f"  Batch size:          {batch_size}")
    print(f"  Max sequence length: {max_sequence_length}")
    print(f"  Stride:              {stride}")
    print(f"  Mask ratio:          {mask_ratio}")

    # Load vector data
    print("\nLoading vector data...")
    parquet_file = '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet'
    df_parquet = pd.read_parquet(parquet_file)

    # Get unique catchment IDs
    catchment_ids = df_parquet['ID'].unique()
    num_catchments = len(catchment_ids)
    print(f"  Found {num_catchments} catchments")

    # Prepare data arrays
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(date_range)

    evap_data = np.zeros((num_catchments, num_days), dtype=np.float32)
    riverflow_data = np.zeros((num_catchments, num_days), dtype=np.float32)

    for idx, catch_id in enumerate(catchment_ids):
        df_catch = df_parquet[df_parquet['ID'] == catch_id].sort_values('date')
        df_catch['date'] = pd.to_datetime(df_catch['date'])
        df_catch = df_catch.set_index('date')
        df_catch = df_catch.reindex(date_range, fill_value=np.nan)

        evap_data[idx, :] = df_catch['evaporation'].values
        riverflow_data[idx, :] = df_catch['discharge_vol'].values

    print(f"  Loaded evap:      {evap_data.shape}")
    print(f"  Loaded riverflow: {riverflow_data.shape}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = MultiModalHydroDataset(
        precip_h5=precip_h5,
        soil_h5=soil_h5,
        temp_h5=temp_h5,
        evap_data=evap_data,
        riverflow_data=riverflow_data,
        static_attr_file=static_attr_file,
        static_attr_vars=static_attr_vars,
        start_date=start_date,
        end_date=end_date,
        max_sequence_length=max_sequence_length,
        stride=stride,
        catchment_ids=catchment_ids,
        stats_cache_path=stats_cache_path,
        land_mask_path=land_mask_path,
        split='train',
        cache_to_memory=True,
    )

    print(f"\n✓ Dataset created: {len(dataset)} samples")

    # Create dataloader
    collate_fn = MultiScaleMaskedCollate(
        max_time_steps=max_sequence_length,
        mask_ratio=mask_ratio,
        land_mask_path=land_mask_path,
        vector_patch_size=8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle to get consistent first batches
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"✓ DataLoader created with batch_size={batch_size}")

    # Get first two batches
    print("\n" + "=" * 80)
    print("  TESTING FIRST TWO BATCHES")
    print("=" * 80)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        print_batch_info(batch_idx + 1, batch)

    print_separator("Verification Complete")
    print("\nKey Checks:")
    print("  ✓ Each batch should have different start dates")
    print("  ✓ All samples in a batch should have the same shapes")
    print("  ✓ Vector modalities should be patchified: [B, num_patches, patch_size, T]")
    print("  ✓ No duplication of the same time window across batches")
    print()


if __name__ == '__main__':
    main()
