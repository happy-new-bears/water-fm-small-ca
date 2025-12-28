"""
Generate merged H5 files for 1970-1988 period (before riverflow data is available)

This script creates:
- precipitation_train_1970_1988.h5
- soil_moisture_train_1970_1988.h5
- temperature_train_1970_1988.h5

These files will allow us to use the 4 modalities (precip, soil, temp, evap)
during 1970-1988 when riverflow data is missing.
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import os


def generate_h5_for_period(
    input_dir: str,
    output_dir: str,
    modality_name: str,
    start_date: str,
    end_date: str,
):
    """
    Generate merged H5 file for specified time period

    Args:
        input_dir: Input directory with monthly h5 files
        output_dir: Output directory
        modality_name: Modality name (e.g., 'precipitation')
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate date list
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_list = []
    current = start
    while current <= end:
        date_list.append(current)
        current += timedelta(days=1)

    num_days = len(date_list)
    print(f"\n{'='*60}")
    print(f"Generating {modality_name} (1970-1988)")
    print(f"{'='*60}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total days: {num_days}")

    # Read first sample to get shape
    first_date = date_list[0]
    first_file = input_path / f"{modality_name}_{first_date.strftime('%Y_%m')}.h5"

    if not first_file.exists():
        raise FileNotFoundError(f"First file not found: {first_file}")

    with h5py.File(first_file, 'r') as f:
        first_key = first_date.strftime('%Y-%m-%d')
        if first_key not in f:
            raise KeyError(f"Date key {first_key} not found in {first_file}")
        sample_shape = f[first_key].shape
        sample_dtype = f[first_key].dtype

    print(f"Image shape: {sample_shape}")
    print(f"Image dtype: {sample_dtype}")

    H, W = sample_shape
    total_size_gb = num_days * H * W * 4 / (1024**3)
    print(f"Estimated output size: {total_size_gb:.2f} GB")

    # Create output file
    output_file = output_path / f"{modality_name}_train_{start_date[:4]}_{end_date[:4]}.h5"

    print(f"\nWriting to: {output_file}")
    print("This may take a few minutes...")

    with h5py.File(output_file, 'w') as f_out:
        # Create dataset with chunking and compression
        data_ds = f_out.create_dataset(
            'data',
            shape=(num_days, H, W),
            dtype=sample_dtype,
            chunks=(30, H, W),  # 30 days per chunk for efficient sequence reading
            compression='gzip',
            compression_opts=4,  # Medium compression level (1-9)
        )

        # Create date index (stored as strings)
        date_strings = [d.strftime('%Y-%m-%d') for d in date_list]
        f_out.create_dataset(
            'dates',
            data=np.array(date_strings, dtype='S10')  # Fixed length string
        )

        # Read by month and write
        current_month = None
        f_in = None

        for day_idx, date in enumerate(tqdm(date_list, desc="Processing")):
            year_month = date.strftime('%Y-%m')

            # Open new file if it's a new month
            if year_month != current_month:
                if f_in is not None:
                    f_in.close()

                input_file = input_path / f"{modality_name}_{date.strftime('%Y_%m')}.h5"

                if not input_file.exists():
                    print(f"\nWarning: File not found: {input_file}")
                    print(f"Filling with zeros for {date.strftime('%Y-%m-%d')}")
                    data_ds[day_idx] = np.zeros((H, W), dtype=sample_dtype)
                    continue

                f_in = h5py.File(input_file, 'r')
                current_month = year_month

            # Read data
            date_key = date.strftime('%Y-%m-%d')

            if date_key in f_in:
                data_ds[day_idx] = f_in[date_key][:]
            else:
                print(f"\nWarning: Date {date_key} not found in file")
                data_ds[day_idx] = np.zeros((H, W), dtype=sample_dtype)

        if f_in is not None:
            f_in.close()

    # Verify output file
    actual_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\n✓ Successfully created: {output_file}")
    print(f"✓ Actual file size: {actual_size_mb:.2f} MB")
    print(f"✓ Compression ratio: {total_size_gb * 1024 / actual_size_mb:.1f}x")

    return output_file


def main():
    """Main function: Generate H5 files for 1970-1988"""

    # Configuration
    data_root = '/Users/transformer/Desktop/water_data/new_version'
    output_root = data_root  # Output to data directory

    # Modality list
    modalities = {
        'precipitation': 'precipitation_processed',
        'soil_moisture': 'soil_moisture_processed',
        'temperature': 'temperature_processed',
    }

    # Time period: 1970-1988
    start_date = '1970-01-01'
    end_date = '1988-12-31'

    print("="*60)
    print("H5 File Generator for 1970-1988 Period")
    print("="*60)
    print(f"Input root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Modalities: {list(modalities.keys())}")

    # Calculate expected days
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    expected_days = (end - start).days + 1
    print(f"Expected days: {expected_days}")

    generated_files = []

    # Generate for each modality
    for modality_name, input_subdir in modalities.items():
        input_dir = f"{data_root}/{input_subdir}"

        # Check if input directory exists
        if not Path(input_dir).exists():
            print(f"\n✗ Directory not found: {input_dir}")
            print("Skipping...")
            continue

        try:
            output_file = generate_h5_for_period(
                input_dir=input_dir,
                output_dir=output_root,
                modality_name=modality_name,
                start_date=start_date,
                end_date=end_date,
            )
            generated_files.append(output_file)
        except Exception as e:
            print(f"\n✗ Error processing {modality_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("✓✓✓ All files generated successfully!")
    print("="*60)
    print(f"\nGenerated files ({len(generated_files)}):")
    for f in generated_files:
        print(f"  - {f.name}")
    print(f"\nFiles are in: {output_root}")
    print("\nNext steps:")
    print("1. Verify the generated files")
    print("2. Update your config to include 1970-1988 period")
    print("3. Implement riverflow_missing flag in dataset and training code")


if __name__ == '__main__':
    main()
