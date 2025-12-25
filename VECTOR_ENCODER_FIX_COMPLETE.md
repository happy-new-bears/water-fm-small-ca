# Vector Encoder Fix - Complete Summary

## 🔥 Critical Bug Fixed

### Problem
**Location**: `models/vector_encoder.py:141` (OLD)

```python
# OLD (WRONG)
visible_patch_mask = ~patch_mask.any(dim=2)  # [B, num_patches]
```

**Impact**:
- Discarded patches if ANY timestep was masked
- Result: **0/76 patches visible** → Encoder receives empty input
- Model **cannot learn** effectively

### Root Cause
The encoder used **per-patch** selection logic (`any(dim=2)`) while the dataset generates **per-(patch, time)** masks:
- Dataset: Each (patch, timestep) pair independently masked at 75%
- Old Encoder: Required ALL timesteps of a patch to be visible
- Probability a patch is NEVER masked across 90 timesteps: `0.25^90 ≈ 6.53e-55` ≈ **0**

### Solution
**Completely rewrite** vector encoder to align with image encoder's per-(position, time) token selection:

```python
# NEW (CORRECT)
visible_mask = ~patch_mask  # [B, num_patches, T]
x_visible = x_aggregated[visible_mask]  # Select ALL visible (patch, time) pairs
num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # [B]
```

**Result**:
- **~1710/6840 tokens visible** (25%) instead of 0
- Encoder receives substantial input
- Model can learn effectively

---

## 🛠️ Implementation Details

### Complete Rewrite of `forward()` Method

The new implementation follows the **exact pattern** from `image_encoder.py`:

#### Step 1: Aggregate Patches
```python
# [B, num_patches, patch_size, T] -> [B, num_patches, T]
if catchment_padding_mask is not None:
    valid_catchment = (~catchment_padding_mask).unsqueeze(-1).float()
    x_aggregated = (x_vec * valid_catchment).sum(dim=2) / (valid_catchment.sum(dim=2) + 1e-6)
    static_aggregated = (static_attr * valid_catchment).sum(dim=2) / (valid_catchment.sum(dim=2) + 1e-6)
else:
    x_aggregated = x_vec.mean(dim=2)
    static_aggregated = static_attr.mean(dim=2)
```

#### Step 2: VECTORIZED Selection of Visible (Patch, Time) Tokens
```python
# Get visibility mask (True = visible)
visible_mask = ~patch_mask  # [B, num_patches, T]

# Select visible tokens using boolean indexing
x_visible = x_aggregated[visible_mask]  # [Total_Visible]

# Calculate number of visible tokens per sample
num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # [B]
max_len = num_visible_per_sample.max().item()
```

#### Step 3: Reshape to [B, max_len] with Padding
```python
if (num_visible_per_sample == max_len).all():
    # FAST PATH: All samples have same length
    x = x_visible.view(B, max_len)
    padding_mask = torch.zeros(B, max_len, device=x_vec.device, dtype=torch.bool)
    lengths = [max_len] * B
else:
    # SLOW PATH: Different lengths, need padding
    x = torch.zeros(B, max_len, device=x_vec.device, dtype=self.in_proj.weight.dtype)
    padding_mask = torch.zeros(B, max_len, device=x_vec.device, dtype=torch.bool)
    lengths = num_visible_per_sample.cpu().tolist()

    offset = 0
    for b in range(B):
        length = lengths[b]
        x[b, :length] = x_visible[offset:offset+length]
        if length < max_len:
            padding_mask[b, length:] = True
        offset += length
```

#### Step 4: Project to d_model
```python
x = self.in_proj(x.unsqueeze(-1))  # [B, max_len, 1] -> [B, max_len, d_model]
```

#### Step 5: VECTORIZED Position Embeddings
**This is the key alignment with image_encoder.py!**

```python
# Create grids of indices [B, num_patches, T]
p_indices = torch.arange(num_patches, device=x_vec.device).view(1, num_patches, 1).expand(B, num_patches, T)
t_indices = torch.arange(T, device=x_vec.device).view(1, 1, T).expand(B, num_patches, T)

# Select indices for visible tokens
p_visible = p_indices[visible_mask].view(B, -1)  # [B, max_len]
t_visible = t_indices[visible_mask].view(B, -1)  # [B, max_len]

# Gather spatial PE
spatial_emb = self.spatial_pos[0, p_visible.view(-1)].view(B, max_len, -1)

# Gather temporal PE
temporal_pe = self.temporal_pos.pe.squeeze(0)
temporal_emb = temporal_pe[t_visible.view(-1)].view(B, max_len, -1)

# Add both PEs (vectorized!)
x = x + spatial_emb + temporal_emb
```

#### Step 6: Add Modality Token
```python
if self.modality_token is not None:
    x = x + self.modality_token  # [1, 1, d_model] broadcast to [B, max_len, d_model]
```

#### Step 7: FiLM Layers with Static Pooling
```python
# Pool static attributes from visible patches
visible_patches = visible_mask.any(dim=2)  # [B, num_patches]
static_pooled = (static_aggregated * visible_patches.unsqueeze(-1).float()).sum(dim=1) / \
                (visible_patches.sum(dim=1, keepdim=True).float() + 1e-6)

# Apply FiLM layers
for layer, film_mlp in zip(self.layers, self.film_mlps):
    gb = film_mlp(static_pooled)
    gamma, beta = gb.chunk(2, dim=-1)
    gamma = gamma.unsqueeze(1)
    beta = beta.unsqueeze(1)
    x = layer(x, gamma, beta, key_padding_mask=padding_mask)
```

#### Step 8: Add Static Token
```python
x = self.norm(x)
static_token = self.attr_proj(static_pooled).unsqueeze(1)
encoder_output = torch.cat([x, static_token], dim=1)

# Update padding mask
static_padding = torch.zeros(B, 1, device=x_vec.device, dtype=torch.bool)
padding_mask_full = torch.cat([padding_mask, static_padding], dim=1)
```

---

## ✅ Verification: Image Encoder vs Vector Encoder

| Aspect | Image Encoder | Vector Encoder | Aligned? |
|--------|---------------|----------------|----------|
| **Mask Logic** | `visible_mask = ~patch_mask` | `visible_mask = ~patch_mask` | ✅ |
| **Token Selection** | `patches[visible_mask]` | `x_aggregated[visible_mask]` | ✅ |
| **Count Visible** | `sum(dim=(1, 2))` | `sum(dim=(1, 2))` | ✅ |
| **Fast Path** | Same length → direct view | Same length → direct view | ✅ |
| **Slow Path** | Loop with offset | Loop with offset | ✅ |
| **Index Grids** | `[B, T, num_valid]` | `[B, num_patches, T]` | ✅ |
| **Index Selection** | Boolean indexing | Boolean indexing | ✅ |
| **PE Gathering** | Vectorized gather | Vectorized gather | ✅ |
| **Modality Token** | After PE | After PE | ✅ |

---

## 📊 Expected Results

### Before Fix
```
Sample 0: 0/76 patches visible (0.00%)
Sample 1: 0/76 patches visible (0.00%)
Sample 2: 0/76 patches visible (0.00%)
Sample 3: 0/76 patches visible (0.00%)
Average: 0.00% ❌
```

### After Fix
```
Sample 0: 1710/6840 tokens visible (25.00%)
Sample 1: 1710/6840 tokens visible (25.00%)
Sample 2: 1710/6840 tokens visible (25.00%)
Sample 3: 1710/6840 tokens visible (25.00%)
Average: 25.00% ✅
```

### Sequence Length Comparison
- **Image modalities**: ~0.25 × 10 × 60 = **~150 tokens**
- **Vector modalities**: ~0.25 × 76 × 90 = **~1710 tokens**
- Both are reasonable and proportional to their input sizes

---

## 🎯 Key Improvements

1. ✅ **No longer discards patches** - Processes all visible (patch, time) pairs
2. ✅ **Sequence length is reasonable** - ~1710 tokens instead of 0
3. ✅ **Completely aligned with Image Encoder** - Same logical pattern
4. ✅ **Vectorized implementation** - No unnecessary loops
5. ✅ **Static token handling** - Correctly pools from visible patches
6. ✅ **Position embeddings** - Vectorized gathering (same as image encoder)

---

## 🚀 Next Steps

1. Run full training to verify the fix resolves the learning issue
2. Monitor validation metrics to confirm model can learn effectively
3. Compare performance with image modalities

---

## 📝 Files Modified

- `models/vector_encoder.py` (Lines 111-305) - Complete rewrite of `forward()` method

---

## 🙏 Credits

- **Teacher**: Identified the critical bug `~patch_mask.any(dim=2)`
- **User**: Correctly observed that Image Encoder also does per-(time, patch) masking
- **Solution**: Complete alignment with Image Encoder's per-(position, time) token selection logic
