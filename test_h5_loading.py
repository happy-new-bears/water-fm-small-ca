"""
简单测试h5文件加载

测试合并后的h5文件是否可以正确读取
"""

import h5py
import numpy as np
import time
from pathlib import Path


def test_h5_file(file_path, modality_name):
    """Test loading a single h5 file"""
    print(f"\n{'='*60}")
    print(f"Testing: {modality_name}")
    print(f"File: {file_path}")
    print(f"{'='*60}")

    if not Path(file_path).exists():
        print(f"❌ File not found!")
        return False

    try:
        # Open file
        with h5py.File(file_path, 'r') as f:
            print(f"✓ File opened successfully")

            # Check keys
            print(f"Keys: {list(f.keys())}")

            # Get data
            data = f['data']
            print(f"Data shape: {data.shape}")
            print(f"Data dtype: {data.dtype}")

            # Test reading a slice
            print(f"\nTesting data access...")
            start_time = time.time()
            sample = data[0:30]  # Read 30 days
            read_time = time.time() - start_time
            print(f"✓ Read 30 days: {sample.shape}")
            print(f"  Read time: {read_time:.4f}s")

            # Test random access
            print(f"\nTesting random access (10 samples)...")
            times = []
            for i in range(10):
                idx = np.random.randint(0, data.shape[0] - 30)
                start_time = time.time()
                sample = data[idx:idx+30]
                times.append(time.time() - start_time)

            avg_time = np.mean(times)
            print(f"✓ Average random read time: {avg_time:.4f}s")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_memory_loading(file_path, modality_name):
    """Test loading entire file to memory"""
    print(f"\n{'='*60}")
    print(f"Testing MEMORY LOADING: {modality_name}")
    print(f"{'='*60}")

    try:
        print(f"Loading entire file to memory...")
        start_time = time.time()

        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]  # Load all to memory

        load_time = time.time() - start_time
        print(f"✓ Loaded to memory in {load_time:.2f}s")
        print(f"  Data shape: {data.shape}")
        print(f"  Memory size: {data.nbytes / (1024**2):.2f} MB")

        # Test access speed
        print(f"\nTesting access speed from memory (100 samples)...")
        times = []
        for i in range(100):
            idx = np.random.randint(0, data.shape[0] - 30)
            start_time = time.time()
            sample = data[idx:idx+30]
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        print(f"✓ Average memory access time: {avg_time*1000:.4f}ms")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test"""
    print("\n" + "="*60)
    print("H5 File Loading Test")
    print("="*60)

    data_root = '/Users/transformer/Desktop/water_data/new_version'

    # Test files
    test_files = {
        'precipitation_train': f'{data_root}/precipitation_train_1989_2010.h5',
        'precipitation_val': f'{data_root}/precipitation_val_2011_2015.h5',
        'soil_train': f'{data_root}/soil_moisture_train_1989_2010.h5',
        'soil_val': f'{data_root}/soil_moisture_val_2011_2015.h5',
        'temperature_train': f'{data_root}/temperature_train_1989_2010.h5',
        'temperature_val': f'{data_root}/temperature_val_2011_2015.h5',
    }

    results = {}

    # Test file access
    print("\n" + "="*60)
    print("PART 1: Testing On-Demand Loading")
    print("="*60)

    for name, path in test_files.items():
        results[name] = test_h5_file(path, name)

    # Test memory loading (only for one file as example)
    print("\n" + "="*60)
    print("PART 2: Testing Memory Caching")
    print("="*60)

    test_memory_loading(
        test_files['precipitation_train'],
        'precipitation_train'
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\nFiles tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")

    if success_count == total_count:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
