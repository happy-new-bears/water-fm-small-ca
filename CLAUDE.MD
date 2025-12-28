# Multi-Modal MAE for Hydrological Forecasting

## Project Overview

This project implements a Multi-Modal Masked Autoencoder (MAE) for hydrological data, combining spatial image data (precipitation, soil moisture, temperature) with vector time series data (evaporation, riverflow) for UK catchments.

### Key Architecture Features

- **Cross-Modal Fusion**: Implements CAV-MAE and CrossMAE architecture for cross-modal learning
- **5 Modalities**: 3 image modalities (precipitation, soil moisture, temperature) + 2 vector modalities (evaporation, riverflow)
- **Spatial Awareness**: Land mask-based spatial encoding with valid patch indices (94 land patches from 290x180 grid)
- **FiLM Integration**: Vector encoders use Feature-wise Linear Modulation to fuse static catchment attributes
- **CrossMAE Decoder**: Uses cross-attention between masked tokens and encoded features instead of traditional self-attention
- **Weighted Feature Maps**: Optional multi-layer encoder feature aggregation (Phase 2)

## Recent Development Timeline

### Latest Commits
1. `1df1cf7` - Fix dtype mismatch in vector encoder FiLM layers
2. `9a64129` - Preserve static tokens in cross-modal fusion
3. `202703f` - Fix critical vector encoder mask selection bug
4. `a803b21` - Implement cross-modal fusion with CAV-MAE and CrossMAE architecture
5. `c9c9a42` - Fix validation mask_ratio inconsistency with training

### Recent Development Focus
- Cross-modal fusion implementation
- Vector encoder bug fixes (mask selection, dtype mismatches)
- Static token preservation in cross-modal context
- Validation consistency improvements

## Project Structure

```
water_fm_small_ca_longer/
├── configs/                      # Configuration files
│   ├── mae_config.py            # Base configuration
│   ├── mae_config_exp1.py       # Experiment 1 config
│   └── mae_config_exp2.py       # Experiment 2 config
├── datasets/                     # Data loading and processing
│   ├── multimodal_dataset_optimized.py  # Main optimized dataset
│   ├── collate.py               # Batch collation with masking
│   ├── data_utils.py            # Data utilities
│   └── fill_missing_values.py   # Handle missing data (NEW)
├── models/                       # Model architecture
│   ├── multimodal_mae.py        # Main MAE model
│   ├── image_encoder.py         # Image modality encoder
│   ├── vector_encoder.py        # Vector modality encoder with FiLM
│   ├── image_decoder.py         # Image modality decoder
│   ├── vector_decoder.py        # Vector modality decoder
│   ├── layers.py                # Shared layers (patchify, etc.)
│   └── spatial_aggregation.py   # Spatial aggregation for vectors
├── utils/                        # Utilities
│   ├── create_spatial_patches.py # Spatial patch creation
│   ├── merge_h5_files.py        # H5 file merging
│   └── nan_detector.py          # NaN detection utility (NEW)
├── train_mae.py                 # Main training script
├── cache/                       # Normalization stats cache
└── output/                      # Model checkpoints and logs
```

## Data Configuration

### Data Sources
- **Image Data**: Pre-processed gridded data (290x180 resolution, 10x10 patches)
  - Precipitation
  - Soil moisture
  - Temperature
- **Vector Data**: 604 catchments with time series data
  - Evaporation
  - Riverflow
- **Static Attributes**: 11 catchment characteristics (latitude, longitude, altitude statistics, area, etc.)

### Data Paths
Located in `/Users/transformer/Desktop/water_data/new_version/`:
- `precipitation_processed/`
- `soil_moisture_processed/`
- `temperature_processed/`
- `riverflow_evaporation_604catchments_1970_2015.parquet`
- `Catchment_attributes/Catchment_attributes_nrfa.csv`
- `gb_temp_valid_mask_290x180.pt`

### Time Periods
- **Training**: 1989-01-01 to 2010-12-31
- **Validation**: 2011-01-01 to 2015-12-30
- **Sequence Length**: 30 days
- **Stride**: 20 days (sliding window)

### Optimized H5 Files
Pre-merged H5 files for faster loading:
- Training: `precipitation_train_1989_2010.h5`, `soil_moisture_train_1989_2010.h5`, `temperature_train_1989_2010.h5`
- Validation: `precipitation_val_2011_2015.h5`, `soil_val_2011_2015.h5`, `temperature_val_2011_2015.h5`

## Model Architecture

### Encoder
- **d_model**: 256 (embedding dimension)
- **Image Encoders**: 6 transformer layers, 8 attention heads
- **Vector Encoders**: 4 transformer layers with FiLM conditioning
- **Dropout**: 0.1

### Decoder
- **decoder_dim**: 128 (smaller than encoder)
- **Decoder Layers**: 4 transformer layers
- **Decoder Type**: CrossAttention (CrossMAE architecture)
- **decoder_self_attn**: False (cross-attention only)

### Masking Strategy
- **Image mask_ratio**: 0.4 (40% of land patches masked)
- **Vector mask_ratio**: 0.4 (40% of timesteps masked)
- **Land threshold**: 0.5 (minimum land coverage for valid patches)
- Only masks land patches (94 valid patches out of 522 total)

### Modality Tokens
- Separate learned tokens for each modality in encoder (d_model=256)
- Separate learned tokens for each modality in decoder (decoder_dim=128)
- Enables cross-modal attention and fusion

### Phase 2 Features (Optional)
- **use_weighted_fm**: Enable weighted feature map aggregation
- **use_fm_layers**: Specify which encoder layers to aggregate
- **use_input**: Include input embeddings as layer 0

## Training Configuration

### Optimization
- **Batch size**: 16 per GPU
- **Learning rate**: 1e-4
- **Weight decay**: 0.05
- **Optimizer**: AdamW (betas=(0.9, 0.95), eps=1e-8)
- **Gradient clipping**: max norm 1.0 (critical for stability)

### Training Schedule
- **Epochs**: 10
- **Warmup epochs**: 5
- **Min LR**: 0.0

### Task Loss Weights
Balances different modality losses:
```python
task_weights = {
    'precip_loss': 100.0,    # Precipitation (small values)
    'soil_loss': 10.0,       # Soil moisture
    'temp_loss': 10.0,       # Temperature
    'evap_loss': 1.0,        # Evaporation (baseline)
    'riverflow_loss': 1.0,   # Riverflow (baseline)
}
```

### Distributed Training (DeepSpeed)
- **ZeRO Stage**: 1
- **Mixed Precision**: FP16
- **Gradient accumulation**: 1 step
- **FP16 Loss Scaling**:
  - Initial scale: 2^16 = 65536
  - Loss scale window: 1000 steps
  - Min loss scale: 1.0
  - Hysteresis: 2

### Performance Optimization
- **Cache images to memory**: True
- **num_workers**: 12
- Pre-merged H5 files for faster I/O

## Running Training

### Basic Usage
```bash
# Single machine, 4 GPUs with default config
deepspeed --num_gpus=4 train_mae.py

# With custom config file
deepspeed --num_gpus=4 train_mae.py --config configs/mae_config_exp1.py

# Using torchrun
torchrun --nproc_per_node=4 train_mae.py --config configs/mae_config_exp1.py
```

### Multiple Experiments in Parallel
```bash
# Experiment 1 on GPUs 0-1
deepspeed --num_gpus=2 --include localhost:0,1 train_mae.py --config configs/mae_config_exp1.py

# Experiment 2 on GPUs 2-3
deepspeed --num_gpus=2 --include localhost:2,3 train_mae.py --config configs/mae_config_exp2.py
```

## Logging and Checkpointing

### Weights & Biases
- **Project**: "water-mae-pretraining"
- **Entity**: None (default)

### Checkpointing
- **Output directory**: `output/multimodal_mae`
- **Checkpoint frequency**: Every 10 epochs
- **Keep last N checkpoints**: 10
- **Log frequency**: Every 10 batches
- **Validation frequency**: Every 1 epoch

## Known Issues and Fixes

### Vector Encoder Fixes
1. **Mask Selection Bug** (202703f): Fixed incorrect mask indexing in vector encoder that was selecting wrong timesteps
2. **FiLM Layer dtype Mismatch** (1df1cf7): Fixed float32/float16 dtype mismatches in FiLM conditioning layers
3. **Static Token Preservation** (9a64129): Ensured static catchment tokens are preserved during cross-modal fusion

### Validation Consistency
- Fixed mask_ratio inconsistency between training and validation (c9c9a42)

## Static Attributes Used

From `Catchment_attributes_nrfa.csv`:
1. latitude
2. longitude
3. minimum-altitude
4. maximum-altitude
5. 50-percentile-altitude
6. 10-percentile-altitude
7. 90-percentile-altitude
8. catchment-area
9. dpsbar (drainage path slope)
10. propwet (proportion of time soils are wet)
11. bfihost (baseflow index)

## Recent Cleanup

Removed temporary development files:
- Analysis scripts (analyze_*.py)
- Test scripts (test_*.py, check_*.py)
- Planning documents (*.md in root)
- Visualization outputs (*.png)
- Temporary installation scripts (install_deepspeed.sh, run_training.sh)

These files were used during development but are not needed in the main codebase.

## Key Implementation Details

### CrossMAE Architecture
- Decoders use cross-attention between masked tokens (queries) and encoded visible tokens (keys/values)
- No self-attention between masked tokens in decoder (follows CrossMAE paper)
- Separate modality tokens for encoder and decoder

### Land Mask Handling
- Only 94 out of 522 patches (290x180 grid → 29x18 patches) are valid land patches
- Valid patch indices stored in model as buffer
- Masking only applies to land patches
- Position embeddings only for valid patches

### Vector Spatial Aggregation
- 604 catchments grouped into spatial patches (vector_patch_size=8)
- Spatial patches: 604 / 8 = 75.5 → 76 patches (last patch padded)
- Enables spatial processing of vector data similar to image patches

## Future Development Considerations

1. **Hyperparameter Tuning**: Experiment with different mask ratios, loss weights, and learning rates
2. **Architecture Variants**: Try different encoder/decoder depths, attention heads
3. **Weighted Feature Maps**: Explore Phase 2 features (use_weighted_fm) for multi-scale representations
4. **Fine-tuning**: Downstream tasks for flood forecasting, drought prediction, etc.
5. **Extended Training**: Current config uses 10 epochs; may need longer training for convergence

## Dependencies

Key libraries used:
- PyTorch (with distributed training)
- DeepSpeed (for efficient distributed training)
- h5py (for HDF5 data loading)
- pandas (for parquet and CSV data)
- numpy
- wandb (optional, for logging)

## Contact and Contribution

This is a research implementation for hydrological forecasting using multi-modal masked autoencoders with cross-modal fusion capabilities.
