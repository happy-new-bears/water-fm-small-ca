# Multi-Modal MAE Mask Strategy

## Overview

This document describes the masking strategy used for Multi-Modal Masked Autoencoder (MAE) pretraining on hydrology data.

## Data Modalities

### Image Modalities (3)
- **Precipitation**: 290×180 spatial grid
- **Soil Moisture**: 290×180 spatial grid
- **Temperature**: 290×180 spatial grid

### Vector Modalities (2)
- **Evaporation**: Single value per timestep
- **Riverflow**: Single value per timestep

### Temporal Dimension
- **Sequence Length**: 90 days (fixed)

---

## Mask Strategy

### 1. IMAGE Masking (Spatial Patch-Level)

**Patch Configuration:**
- Patch size: 10×10 pixels
- Total patches: 29×18 = 522 patches
- Valid patches: 94 patches (land coverage ≥50%)
- Ocean patches: 428 patches (excluded from training)

**Masking Strategy:**
```
Each day independently masks 75% of valid land patches

Day 1: Randomly mask 70 patches (out of 94 valid)
Day 2: Randomly mask 70 patches (different from Day 1)
Day 3: Randomly mask 70 patches (different from Day 1 & 2)
...
Day 90: Randomly mask 70 patches
```

**Key Properties:**
- ✓ Mask ratio: 75% of valid patches per day
- ✓ Each day has different masked patches (spatial variation)
- ✓ Every day has some visible information (24 patches)
- ✓ Ocean patches are NEVER masked (not used in training)

**Visual Pattern:**
```
Time →
Patch 1:  [V] [M] [V] [M] [V] ...
Patch 2:  [M] [V] [M] [M] [V] ...
Patch 3:  [V] [M] [V] [V] [M] ...
...
Patch 94: [M] [M] [V] [M] [V] ...

[V] = Visible (encoder input)
[M] = Masked (decoder predicts)
```

---

### 2. VECTOR Masking (Temporal)

**Masking Strategy:**
```
Randomly select 75% of days to mask entirely

Masked days: 67 days (entire day masked)
Visible days: 23 days (entire day visible)
```

**Key Properties:**
- ✓ Mask ratio: 75% of total days
- ✓ Masked days are randomly selected
- ✓ Entire day is either masked or visible (no partial masking)
- ⚠️ Creates temporal gaps in vector data

**Visual Pattern:**
```
Time →
Evap:      [M] [M] [V] [M] [M] [V] [M] [M] [M] [V] ...
Riverflow: [M] [M] [V] [M] [M] [V] [M] [M] [M] [V] ...

[V] = Visible (encoder input)
[M] = Masked (decoder predicts)
```

---

## Implementation Details

### Land Mask Processing

The land mask (`gb_temp_valid_mask_290x180.pt`) is used to identify valid patches:

```python
Valid Patch Criteria:
- Land coverage ≥ 50% (threshold=0.5)
- Total: 94 out of 522 patches (18%)

Statistics:
- Fully ocean: 369 patches (70.7%) - never used
- Partially land: 99 patches (19.0%)
- Fully land: 54 patches (10.3%)
```

### Mask Generation

**Image Mask:**
```python
Shape: [B, T=90, num_patches=522]

For each batch sample:
    For each timestep (day):
        1. Initialize mask: all False [522]
        2. Randomly select 70 patches from 94 valid patches
        3. Set mask[selected_patches] = True

Result: [B, 90, 522] boolean tensor
```

**Vector Mask:**
```python
Shape: [B, T=90]

For each batch sample:
    1. Initialize mask: all False [90]
    2. Randomly select 67 days from 90 days
    3. Set mask[selected_days] = True

Result: [B, 90] boolean tensor
```

---

## MAE Training Process

### 1. Encoder Stage

**Image:**
```python
1. Patchify images: [B, T, 290, 180] → [B, T, 522, 100]
2. Remove masked patches: keep only visible patches
3. Encoder processes: [B, T, ~24, 100] (only visible patches)
```

**Vector:**
```python
1. Take vector data: [B, T, 2]
2. Remove masked timesteps: keep only visible days
3. Encoder processes: [B, ~23, 2] (only visible days)
```

**Key Point:** Encoder never sees masked data (they are removed, not set to 0)

### 2. Decoder Stage

```python
1. Decoder receives encoder output
2. Insert learnable mask_token at masked positions
3. Decoder reconstructs full sequence
4. Loss computed ONLY on masked positions
```

### 3. Loss Calculation

```python
Image Loss:
- Compare: predicted[masked] vs original[masked]
- Only masked patches contribute to loss
- ~70 patches per day × 90 days

Vector Loss:
- Compare: predicted[masked] vs original[masked]
- Only masked days contribute to loss
- ~67 days × 2 variables
```

---

## Mask Ratio Statistics

### Overall Mask Coverage

**Image:**
```
Total elements: 90 days × 522 patches = 46,980
Masked elements: 90 days × 70 patches = 6,300
Mask ratio: 6,300 / 46,980 = 13.4%
(Only considering valid patches: 70/94 = 74.5%)
```

**Vector:**
```
Total elements: 90 days
Masked elements: 67 days
Mask ratio: 67 / 90 = 74.4%
```

---

## Visualization

See `mask_visualization_with_legend.png` for visual representation of:
- Image mask pattern (showing temporal and spatial distribution)
- Vector mask pattern (showing masked vs visible days)
- Color legend (red=masked, green=visible)

---

## Configuration

```python
from datasets import MultiScaleMaskedCollate

# Training mode
train_collate = MultiScaleMaskedCollate(
    seq_len=90,                    # Fixed sequence length
    mask_ratio=0.75,               # Mask 75%
    patch_size=10,                 # 10×10 patches
    land_mask_path='path/to/land_mask.pt',
    land_threshold=0.5,            # Valid if ≥50% land
    mask_mode='unified',           # Same mask for all image modalities
    mode='train'
)

# Validation mode (no masking)
val_collate = MultiScaleMaskedCollate(
    seq_len=90,
    land_mask_path='path/to/land_mask.pt',
    land_threshold=0.5,
    mode='val'                     # Disables all masking
)
```

---

## References

- Original MAE paper: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- MAE official implementation: [facebookresearch/mae](https://github.com/facebookresearch/mae)
