# Vector Encoder 修改计划

## 问题

当前 Vector Encoder 使用了错误的逻辑：
```python
visible_patch_mask = ~patch_mask.any(dim=2)  # 错误！丢弃了大部分 patches
```

这导致：
- **0/76 patches 可见**（因为每个 patch 至少有一个时间步被 mask）
- **Encoder 输入几乎为空**
- **模型无法学习**

## 正确的逻辑（对齐 Image Encoder）

应该选择**所有可见的 (patch, time) 对**，而不是"完全可见的 patches"。

### Image Encoder 的逻辑（参考）

```python
# patch_mask: [B, T, num_patches]
visible_mask = ~patch_mask  # [B, T, num_patches] - True = visible
x_visible = patches[visible_mask]  # 选择所有可见的 (t, p) tokens
num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # 统计每个样本的可见 token 数
```

###Vector Encoder 应该一模一样

```python
# patch_mask: [B, num_patches, T]
visible_mask = ~patch_mask  # [B, num_patches, T] - True = visible
x_visible = patches[visible_mask]  # 选择所有可见的 (p, t) tokens
num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # 统计每个样本的可见 token 数
```

## 详细修改步骤

### Step 1: Aggregate patches
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

### Step 2: Select visible (patch, time) tokens
```python
# 完全对齐 Image Encoder
visible_mask = ~patch_mask  # [B, num_patches, T]
x_visible = x_aggregated[visible_mask]  # [Total_Visible]
num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # [B]
max_len = num_visible_per_sample.max().item()
```

### Step 3: Reshape to [B, max_len]
```python
if (num_visible_per_sample == max_len).all():
    # FAST PATH
    x = x_visible.view(B, max_len)
    padding_mask = torch.zeros(B, max_len, device=x_vec.device, dtype=torch.bool)
    lengths = [max_len] * B
else:
    # SLOW PATH (with padding)
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

### Step 4: Project to d_model
```python
x = self.in_proj(x.unsqueeze(-1))  # [B, max_len, 1] -> [B, max_len, d_model]
```

### Step 5: VECTORIZED position embeddings
```python
# Create index grids
p_indices = torch.arange(num_patches, device=x_vec.device).view(1, num_patches, 1).expand(B, num_patches, T)
t_indices = torch.arange(T, device=x_vec.device).view(1, 1, T).expand(B, num_patches, T)

# Select for visible tokens
p_visible = p_indices[visible_mask].view(B, -1)  # [B, max_len]
t_visible = t_indices[visible_mask].view(B, -1)  # [B, max_len]

# Gather PEs
spatial_emb = self.spatial_pos[0, p_visible.view(-1)].view(B, max_len, -1)
temporal_emb = self.temporal_pos.pe.squeeze(0)[t_visible.view(-1)].view(B, max_len, -1)

# Add PEs
x = x + spatial_emb + temporal_emb
```

### Step 6: Add modality token
```python
if self.modality_token is not None:
    x = x + self.modality_token
```

### Step 7: FiLM layers (保持不变)
```python
# 计算全局 static vector
# ... (当前逻辑保持不变)
```

## 关键改进

1. ✅ **不再丢弃 patches** - 处理所有可见的 (patch, time) 对
2. ✅ **序列长度合理** - ~0.25 × num_patches × T ≈ 0.25 × 76 × 90 ≈ 1710（与 Image 一致）
3. ✅ **完全对齐 Image Encoder** - 逻辑完全一致
4. ✅ **Vectorized 实现** - 无不必要的循环

## 测试预期

修改后，使用 `test_mask_mismatch.py` 测试应该显示：
- ✅ Visible patches: ~19/76 (25%) 而不是 0/76
- ✅ Encoder 输入: ~1710 tokens 而不是 0
- ✅ 模型可以正常学习
