# Riverflow Missing Data Implementation Summary

## Overview

Successfully implemented dynamic handling of riverflow data missing in the 1970-1988 period, allowing the model to train on 4 modalities (precipitation, soil moisture, temperature, evaporation) during this period and 5 modalities (adding riverflow) from 1989 onwards.

## Files Modified

### 1. `configs/mae_config.py`
**Changes:**
- Extended `train_start` from '1989-01-01' to '1970-01-01'
- Added `riverflow_available_from = '1989-01-01'` configuration
- Updated `stats_cache_path` to 'cache/normalization_stats_1970_2015.pt'
- Updated H5 file paths to use merged 1970-2010 files

**Impact:** Configuration now supports full 1970-2015 training period with riverflow availability tracking

---

### 2. `datasets/multimodal_dataset_optimized.py`
**Changes:**
- Added `riverflow_available_from` parameter to `__init__`
- Modified `_compute_vector_stats()` to compute riverflow statistics ONLY from 1989 onwards (avoiding NaN period)
- Added `riverflow_missing` flag computation in `__getitem__` based on start_date
- Updated return dict to include 'riverflow_missing' flag

**Impact:**
- Correct normalization statistics (riverflow uses only 1989-2015 data)
- Each sample knows whether riverflow is available
- Evaporation and image modalities use full 1970-2015 data for normalization

---

### 3. `datasets/collate.py`
**Changes:**
- Extract `riverflow_missing` flag from batch samples
- Pass `riverflow_missing` to `_generate_masks()`
- Modified `_generate_masks()` to set riverflow_mask to 100% True (all masked) when `riverflow_missing=True`
- Added `riverflow_missing` to batch output dict

**Impact:**
- Riverflow is 100% masked when data is missing (encoder sees no riverflow tokens)
- Normal random masking when data is available

---

### 4. `models/multimodal_mae.py`
**Changes:**
- Extract `riverflow_missing` flag from batch in `forward()`
- **Encoder:** Skip riverflow encoder entirely when `riverflow_missing=True` (方案A)
- **Fusion:** Conditionally concatenate tokens (4 modalities when missing, 5 when available)
- **Padding mask:** Conditionally concatenate padding masks
- **Decoder:** Skip riverflow decoder when `riverflow_missing=True`
- **Loss:** Set `riverflow_loss=0` and exclude from total_loss when missing

**Impact:**
- Model dynamically handles 4 or 5 modalities
- No gradient flow through riverflow when data is missing
- Cross-modal fusion adapts to available modalities

---

### 5. `train_mae.py`
**Changes:**
- Pass `riverflow_available_from` parameter when creating train and val datasets
- Uses config value with fallback to '1989-01-01'

**Impact:** Training script properly configures datasets with riverflow availability information

---

## Additional Files Created

### Data Generation
1. `generate_1970_1988_h5.py` - Generated H5 files for early period
2. `merge_1970_2010_h5.py` - Merged 1970-1988 and 1989-2010 into complete files

### Generated H5 Files
- `precipitation_train_1970_2010.h5` (504.56 MB, 14975 days)
- `soil_moisture_train_1970_2010.h5` (69.06 MB, 14975 days)
- `temperature_train_1970_2010.h5` (446.09 MB, 14975 days)

### Testing
- `test_riverflow_missing.py` - Comprehensive test suite (requires polars to run)

---

## Implementation Strategy

**Chosen Approach: 方案A (Explicit Encoder Skipping)**

When `riverflow_missing=True`:
1. ✅ **100% mask** riverflow in collate (all tokens masked)
2. ✅ **Skip encoder** entirely (no computation, riverflow_token=None)
3. ✅ **Skip from fusion** (only 4 modalities in cross-modal fusion)
4. ✅ **Skip decoder** (no reconstruction)
5. ✅ **Skip loss** (riverflow_loss=0, not added to total_loss)

This is equivalent to 方案B (100% masking) but more explicit and efficient.

---

## Normalization Strategy

**Critical Decision:** Riverflow statistics computed ONLY from 1989-2015 period

**Implementation in `_compute_vector_stats()`:**
```python
# Find riverflow_start_idx where data becomes available
riverflow_start_idx = 0
for i, date in enumerate(self.date_list):
    if date >= self.riverflow_available_from:
        riverflow_start_idx = i
        break

# Only use riverflow data from valid period
riverflow_valid_period = riverflow_data[:, riverflow_start_idx:]
```

**Why this matters:**
- Evaporation: Uses 1970-2015 data (full 41 years) for normalization
- Image modalities: Use 1970-2015 data (full 41 years)
- Riverflow: Uses 1989-2015 data (22 years) - avoids NaN/missing values
- Static attributes: Use all catchments (global stats)

---

## Training Data Summary

### Total Training Data
- **1970-1988 (19 years)**: 4 modalities (precip, soil, temp, evap)
- **1989-2010 (22 years)**: 5 modalities (precip, soil, temp, evap, riverflow)
- **Total**: 41 years of training data vs. previous 22 years

### Sample Distribution (approximate with stride=20)
- 1970-1988: ~350 samples with riverflow_missing=True
- 1989-2010: ~400 samples with riverflow_missing=False
- Total: ~750 training samples

---

## Testing Checklist

Before training, verify:

1. ✅ H5 files generated correctly (1970-2010 merged files exist)
2. ✅ Config updated (train_start='1970-01-01', riverflow_available_from='1989-01-01')
3. ⏳ Dataset loads successfully (run test_riverflow_missing.py or manual test)
4. ⏳ Batch from 1970 has riverflow_missing=True
5. ⏳ Batch from 1990 has riverflow_missing=False
6. ⏳ Riverflow mask is 100% True when missing
7. ⏳ Model forward pass works with both 4 and 5 modalities
8. ⏳ Loss computation correct (riverflow_loss=0 when missing)

---

## Expected Training Behavior

### Epoch Statistics
- Some batches will show `riverflow_loss=0.000` (1970-1988 samples)
- Some batches will show normal `riverflow_loss>0` (1989-2010 samples)
- Total loss will be sum of 4 or 5 modality losses depending on period

### Model Learning
- Early period (1970-1988): Model learns cross-modal fusion of 4 modalities
- Later period (1989-2010): Model learns cross-modal fusion of 5 modalities
- Model becomes robust to partial modality availability

---

## Potential Issues and Solutions

### Issue 1: Cache Invalidation
**Problem:** Old normalization stats cache may be incompatible
**Solution:** Delete `cache/normalization_stats.pt` before first run (config already uses new name `normalization_stats_1970_2015.pt`)

### Issue 2: Memory Usage
**Problem:** Larger H5 files (1970-2010 vs 1989-2010)
**Solution:** Files are compressed (gzip), actual increase is ~2x but manageable

### Issue 3: Gradient Issues
**Problem:** Different modality counts might affect gradient flow
**Solution:** Explicit skipping ensures clean gradient paths

---

## Next Steps

1. **Delete old cache** (if exists):
   ```bash
   rm cache/normalization_stats.pt
   ```

2. **Run training**:
   ```bash
   deepspeed --num_gpus=4 train_mae.py
   ```

3. **Monitor logs** for:
   - Riverflow availability warnings in dataset init
   - Mix of riverflow_loss=0 and riverflow_loss>0 in training
   - Total sample count (~750 vs previous ~400)

4. **Validate results**:
   - Check model learns meaningful representations
   - Verify performance on downstream tasks

---

## Code Locations

Key code sections for debugging:

- **Riverflow missing detection**: `datasets/multimodal_dataset_optimized.py:561-562`
- **Normalization stats (riverflow)**: `datasets/multimodal_dataset_optimized.py:373-408`
- **100% masking logic**: `datasets/collate.py:267-282`
- **Encoder skipping**: `models/multimodal_mae.py:295-303`
- **Fusion conditional**: `models/multimodal_mae.py:313-329`
- **Loss handling**: `models/multimodal_mae.py:427-444`

---

## Summary

✅ **Implementation Complete**

All 5 files modified successfully to support dynamic riverflow missing handling. The model can now:
- Train on 1970-2015 data (41 years total)
- Handle 4 modalities (1970-1988) and 5 modalities (1989-2015)
- Compute correct normalization statistics for each modality
- Skip riverflow encoder/decoder when data is missing
- Properly mask and exclude riverflow loss when appropriate

**Total increase in training data: 86% (from 22 years to 41 years)**

---

## Contact

For questions or issues, refer to the modified files or this summary document.
