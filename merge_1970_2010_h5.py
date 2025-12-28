"""
Merge 1970-1988 and 1989-2010 H5 files into complete 1970-2010 files
"""

import h5py
import numpy as np
from datetime import datetime

def merge_two_h5_files(early_file, later_file, output_file, modality_name):
    """Merge two H5 files into one"""

    print(f"\n{'='*60}")
    print(f"Merging {modality_name}")
    print(f"{'='*60}")
    print(f"Early file: {early_file}")
    print(f"Later file: {later_file}")
    print(f"Output: {output_file}")

    # Read early file
    with h5py.File(early_file, 'r') as f_early:
        data_early = f_early['data'][:]
        dates_early = f_early['dates'][:]
        print(f"\nEarly period:")
        print(f"  Shape: {data_early.shape}")
        print(f"  Dates: {dates_early[0]} to {dates_early[-1]}")

    # Read later file
    with h5py.File(later_file, 'r') as f_later:
        data_later = f_later['data'][:]
        dates_later = f_later['dates'][:]
        print(f"\nLater period:")
        print(f"  Shape: {data_later.shape}")
        print(f"  Dates: {dates_later[0]} to {dates_later[-1]}")

    # Merge
    data_merged = np.concatenate([data_early, data_later], axis=0)
    dates_merged = np.concatenate([dates_early, dates_later], axis=0)

    print(f"\nMerged:")
    print(f"  Shape: {data_merged.shape}")
    print(f"  Dates: {dates_merged[0]} to {dates_merged[-1]}")

    # Calculate expected days
    expected_days = (datetime(2010, 12, 31) - datetime(1970, 1, 1)).days + 1
    print(f"  Expected days (1970-01-01 to 2010-12-31): {expected_days}")

    if data_merged.shape[0] == expected_days:
        print(f"  ✅ Day count is CORRECT!")
    else:
        print(f"  ❌ Day count MISMATCH! Got {data_merged.shape[0]}, expected {expected_days}")
        return

    # Write merged file
    print(f"\nWriting merged file...")
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset(
            'data',
            data=data_merged,
            chunks=(30, 290, 180),
            compression='gzip',
            compression_opts=4,
        )
        f_out.create_dataset(
            'dates',
            data=dates_merged
        )

    import os
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✅ Successfully created: {output_file}")
    print(f"✅ File size: {file_size_mb:.2f} MB")


def main():
    data_root = '/Users/transformer/Desktop/water_data/new_version'

    modalities = [
        ('precipitation',
         f'{data_root}/precipitation_train_1970_1988.h5',
         f'{data_root}/precipitation_train_1989_2010.h5',
         f'{data_root}/precipitation_train_1970_2010.h5'),
        ('soil_moisture',
         f'{data_root}/soil_moisture_train_1970_1988.h5',
         f'{data_root}/soil_moisture_train_1989_2010.h5',
         f'{data_root}/soil_moisture_train_1970_2010.h5'),
        ('temperature',
         f'{data_root}/temperature_train_1970_1988.h5',
         f'{data_root}/temperature_train_1989_2010.h5',
         f'{data_root}/temperature_train_1970_2010.h5'),
    ]

    print("="*60)
    print("Merging 1970-1988 and 1989-2010 H5 Files")
    print("="*60)

    for modality_name, early_file, later_file, output_file in modalities:
        try:
            merge_two_h5_files(early_file, later_file, output_file, modality_name)
        except Exception as e:
            print(f"\n❌ Error merging {modality_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ All files merged successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
