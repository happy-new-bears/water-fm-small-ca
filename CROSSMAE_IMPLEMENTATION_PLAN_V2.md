# ⚠️ CrossMAE改造计划 - 重要架构差异说明

## 🔴 关键发现：架构根本差异

### 当前架构 (water_fm_small):
```python
Image Encoder:
[B, T, H, W]
  → patchify → [B, T, 522, patch_dim]
  → filter valid → [B, T, 94, patch_dim]
  → remove masked → [B, L_visible_total, patch_dim]  # L_visible_total ≈ 2160
  → transformer → [B, L_visible_total, d_model]
  → **POOLING** → [B, d_model]  ← 关键：pool成单个token!

Image Decoder:
encoder_token [B, d_model]
  → 创建完整序列 [visible + masked]
  → self-attention
  → 预测所有位置
```

### CrossMAE架构:
```python
Image Encoder:
[B, 3, H, W]
  → patchify → [B, L, patch_dim]  # L = 196
  → remove masked → [B, L_visible, patch_dim]  # L_visible ≈ 49
  → transformer → [B, L_visible, d_model]
  → **NO POOLING!** 保留序列 [B, L_visible, d_model]  ← 关键差异!

Image Decoder:
visible_tokens [B, L_visible, d_model]  # 保留序列！
  → 只创建masked queries [B, L_masked, d_model]
  → cross-attention: queries attend to all visible tokens
  → 预测masked位置
```

**核心差异：CrossMAE的encoder保留了序列维度，没有pool成单个token！**

---

## 🎯 两种实现方案

### 方案A：轻量级改造（快速实现，但不完全符合CrossMAE）

**改动**：
- ✅ Encoder保持现有架构，pool成 [B, d_model]
- ✅ Decoder用CrossAttention，但keys/values是单个token
- ❌ 所有masked positions看到相同的context (失去细粒度信息)

**优点**：
- 改动最小
- 实现快速
- 能获得部分加速

**缺点**：
- 不是真正的CrossMAE
- 没有发挥CrossMAE的核心优势（masked attend to 多个visible tokens）
- 性能提升有限

---

### 方案B：完整CrossMAE（推荐，彻底改造）⭐

**改动**：
- 🔄 **重大改动**：Encoder不再pool，保留序列 [B, L_visible, d_model]
- 🔄 Decoder创建masked queries，attend to所有visible tokens
- 🔄 需要修改multimodal_mae.py中的fusion逻辑

**优点**：
- ✅ 完全符合CrossMAE精神
- ✅ Masked positions可以看到所有visible positions的细节
- ✅ 发挥CrossMAE的核心优势
- ✅ 预期性能最佳

**缺点**：
- 改动较大（但值得）
- 需要重新设计modality fusion

---

## 📋 采用方案B的详细实施计划

### 🔧 Phase 0: 架构调整（新增，最关键！）

#### Step 0.1: 修改Image Encoder - 移除Pooling
**文件**: `models/image_encoder.py`

**当前代码**:
```python
def forward(self, x_img, patch_mask):
    # ... patchify, filter, position encoding

    # Transformer
    x = self.transformer(x, src_key_padding_mask=padding_mask)

    # ❌ Pooling - 要移除！
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    encoder_token = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
    encoder_token = self.norm(encoder_token)  # [B, d_model]

    return encoder_token, mask_info
```

**新代码（CrossMAE风格）**:
```python
def forward(self, x_img, patch_mask):
    """
    Returns:
        encoder_output: [B, L_visible, d_model] 保留序列！
        mask_info: dict
    """
    # ... patchify, filter, position encoding

    # Transformer
    x = self.transformer(x, src_key_padding_mask=padding_mask)

    # ✅ 不pool！保留序列
    # 只做normalization
    x = self.norm(x)  # [B, L_visible, d_model]

    mask_info = {
        'mask': patch_mask,
        'padding_mask': padding_mask,  # 重要：传递给decoder
        'positions': positions_padded,
    }

    return x, mask_info  # [B, L_visible, d_model] - 保留序列！
```

**关键改动**:
1. ❌ 删除pooling操作
2. ✅ 返回完整的序列 [B, L_visible, d_model]
3. ✅ 在mask_info中传递padding_mask（decoder需要）

---

#### Step 0.2: 修改Image Decoder - 接收序列
**文件**: `models/image_decoder.py`

**当前代码**:
```python
def forward(self, encoder_token, mask_info):
    # encoder_token: [B, d_model] 单个token

    # 创建完整序列
    x = self.mask_token.expand(B, T, num_patches, -1)
    x[visible_mask] = encoder_token  # broadcast到所有位置

    # Self-attention
    x = self.transformer(x)
    return pred_patches
```

**新代码（CrossMAE风格）**:
```python
def forward(self, encoder_output, mask_info):
    """
    Args:
        encoder_output: [B, L_visible, d_model] - 序列！
        mask_info: dict with 'mask', 'padding_mask', etc.

    Returns:
        pred_patches: [B, T, num_patches, patch_dim]
    """
    mask = mask_info['mask']  # [B, T, num_patches]
    padding_mask = mask_info.get('padding_mask')  # [B, L_visible]

    # Step 1: 创建masked queries（参考CrossMAE mask_tokens_grid）
    masked_queries = []
    masked_positions = []  # (b, t, patch_idx)

    for b in range(B):
        for t in range(T):
            for p in range(num_patches):
                if mask[b, t, p]:  # True = masked
                    # Query = mask_token + positional embeddings
                    query = self.mask_token.squeeze() + \
                            self.spatial_pos[0, 0, p] + \
                            self.temporal_pos.pe[0, t]
                    masked_queries.append(query)
                    masked_positions.append((b, t, p))

    if len(masked_queries) == 0:
        return torch.zeros(B, T, num_patches, patch_dim, device=mask.device)

    queries = torch.stack(masked_queries)  # [total_masked, decoder_dim]

    # Step 2: Prepare keys/values from encoder
    # encoder_output: [B, L_visible, encoder_dim]
    # 需要expand给每个query

    # 方式1: 把所有batch的visible tokens拼在一起
    keys_values_list = []
    for b in range(B):
        if padding_mask is not None:
            # 只取非padding的位置
            valid_mask = ~padding_mask[b]  # [L_visible]
            valid_tokens = encoder_output[b, valid_mask]  # [L_valid, encoder_dim]
        else:
            valid_tokens = encoder_output[b]  # [L_visible, encoder_dim]
        keys_values_list.append(valid_tokens)

    # 方式2: 对每个query，attend to 对应batch的visible tokens
    # 需要为每个query记录其所属的batch

    # Step 3: CrossAttention decoder blocks
    x = queries.unsqueeze(0)  # [1, total_masked, decoder_dim]

    for blk in self.decoder_blocks:
        # 这里需要特殊处理：每个query只attend to自己batch的visible tokens
        # 方式A: 使用attention mask
        # 方式B: 分batch处理
        x = blk(x, keys_values)  # CrossAttentionBlock

    # Step 4: Prediction
    x = self.decoder_norm(x)
    predictions = self.pred_head(x)  # [1, total_masked, patch_dim]

    # Step 5: 重组
    pred_patches = torch.zeros(B, T, num_patches, patch_dim, device=mask.device)
    for idx, (b, t, p) in enumerate(masked_positions):
        pred_patches[b, t, p] = predictions[0, idx]

    return pred_patches
```

**但是！这里有个问题**：
- 每个query需要attend to**自己batch**的visible tokens
- 不是attend to所有batch的tokens
- 需要用attention mask或者分batch处理

**更好的实现（分batch）**:
```python
def forward(self, encoder_output, mask_info):
    """
    Args:
        encoder_output: [B, L_visible, d_model]
        mask_info: dict

    Returns:
        pred_patches: [B, T, num_patches, patch_dim]
    """
    mask = mask_info['mask']  # [B, T, num_patches]
    padding_mask = mask_info.get('padding_mask')  # [B, L_visible]
    B, T, num_patches = mask.shape

    pred_patches = torch.zeros(B, T, num_patches, self.patch_dim,
                                device=mask.device, dtype=encoder_output.dtype)

    # 逐batch处理（更简单，更清晰）
    for b in range(B):
        # 1. 获取这个batch的visible tokens
        if padding_mask is not None:
            valid_mask = ~padding_mask[b]
            keys_values = encoder_output[b:b+1, valid_mask]  # [1, L_valid, D]
        else:
            keys_values = encoder_output[b:b+1]  # [1, L_visible, D]

        # 2. 创建这个batch的masked queries
        batch_queries = []
        batch_positions = []  # (t, p)

        for t in range(T):
            for p in range(num_patches):
                if mask[b, t, p]:
                    query = self.mask_token.squeeze() + \
                            self.spatial_pos[0, 0, p] + \
                            self.temporal_pos.pe[0, t]
                    batch_queries.append(query)
                    batch_positions.append((t, p))

        if len(batch_queries) == 0:
            continue

        queries = torch.stack(batch_queries).unsqueeze(0)  # [1, L_masked_b, D]

        # 3. CrossAttention decoder
        x = queries
        for blk in self.decoder_blocks:
            x = blk(x, keys_values)  # queries attend to this batch's visible

        # 4. Prediction
        x = self.decoder_norm(x)
        predictions = self.pred_head(x)  # [1, L_masked_b, patch_dim]

        # 5. Fill predictions
        for idx, (t, p) in enumerate(batch_positions):
            pred_patches[b, t, p] = predictions[0, idx]

    return pred_patches
```

---

### 🔧 Phase 1: CrossAttention模块实现

（保持不变，参考原计划）

---

### 🔧 Phase 2: WeightedFeatureMaps（可选）

#### 关键改动：保存多层的**序列输出**

**Image Encoder with WeightedFeatureMaps**:
```python
def forward(self, x_img, patch_mask):
    # ... patchify, filter, position encoding

    if self.use_weighted_fm:
        x_feats = []

        if self.use_input:
            x_feats.append(x.clone())  # [B, L_visible, d_model]

        for idx, blk in enumerate(self.transformer.layers):
            x = blk(x, src_key_padding_mask=padding_mask)

            if idx in self.use_fm_layers:
                x_feats.append(self.norm(x.clone()))  # 保存序列！

        # 返回list of [B, L_visible, d_model]
        return x_feats, mask_info
    else:
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x, mask_info  # [B, L_visible, d_model]
```

**Image Decoder with WeightedFeatureMaps**:
```python
def forward(self, encoder_output, mask_info):
    # encoder_output: list of [B, L_visible, d_model]

    # 对每个batch的每个masked position:
    for b in range(B):
        if padding_mask is not None:
            valid_mask = ~padding_mask[b]

        # Keys/Values: 来自多层encoder
        if self.use_weighted_fm:
            # 提取这个batch的所有层的visible tokens
            kv_layers = []
            for feat in encoder_output:  # list
                if padding_mask is not None:
                    kv_layers.append(feat[b:b+1, valid_mask])  # [1, L_valid, D]
                else:
                    kv_layers.append(feat[b:b+1])

            # WeightedFeatureMaps: 组合多层
            # 需要在 [1, L_valid, D, num_layers] 上操作
            stacked_kv = torch.stack(kv_layers, dim=-1)  # [1, L_valid, D, k]
            weighted_kv = self.wfm(stacked_kv)  # [1, L_valid, D, decoder_depth]
        else:
            keys_values = encoder_output[b:b+1, valid_mask]

        # ... 创建queries

        # CrossAttention with weighted features
        x = queries
        for i, blk in enumerate(self.decoder_blocks):
            if self.use_weighted_fm:
                # 第i个decoder层用第i个feature组合
                kv_i = self.dec_norms[i](weighted_kv[..., i])  # [1, L_valid, D]
                x = blk(x, kv_i)
            else:
                x = blk(x, keys_values)
```

---

## 🚨 需要额外处理的问题

### 问题1: Modality Fusion
**当前**: 每个modality的encoder输出 [B, d_model]，可以直接concat或add
```python
fused = torch.cat([precip_token, soil_token, temp_token, ...], dim=-1)
```

**CrossMAE风格**: 每个modality输出 [B, L_visible, d_model]
- L_visible每个modality都不同！
- 如何fusion？

**解决方案**:
1. **Option A**: 每个modality分别decoder，不做fusion
2. **Option B**: Fusion时先pool各modality，然后用fused token作为额外的keys/values
3. **Option C**: 在decoder中做cross-modality attention

推荐：**Option A** (简单) 或 **Option B** (保留fusion能力)

---

### 问题2: Static Attributes
**当前**: Static attributes与encoder token concat
```python
encoder_token = torch.cat([encoder_token, static_attr], dim=-1)
```

**CrossMAE风格**: Encoder输出是序列 [B, L_visible, d_model]

**解决方案**:
```python
# Option A: Static attr加到每个visible token上
static_expanded = static_attr.unsqueeze(1).expand(-1, L_visible, -1)
encoder_output = torch.cat([encoder_output, static_expanded], dim=-1)

# Option B: Static attr作为额外的token
static_token = self.static_proj(static_attr).unsqueeze(1)  # [B, 1, d_model]
encoder_output = torch.cat([encoder_output, static_token], dim=1)  # [B, L_visible+1, d_model]
```

推荐：**Option B** (更符合transformer风格)

---

## ✅ 最终实施步骤

### Phase 0: 架构调整（新增）⭐
1. [x] 修改Image Encoder移除pooling
2. [x] 修改Image Decoder接收序列并实现逐batch处理
3. [x] 修改Vector Encoder移除pooling
4. [x] 修改Vector Decoder接收序列
5. [x] 调整Static Attributes处理
6. [x] 测试基础功能（不用CrossAttention，只是架构改变）

### Phase 1: CrossAttention
7. [x] 实现CrossAttention和CrossAttentionBlock
8. [x] 替换Decoder中的self-attention为cross-attention
9. [x] 测试CrossAttention版本

### Phase 2: WeightedFeatureMaps（可选）
10. [x] 实现WeightedFeatureMaps
11. [x] 修改Encoder保存多层序列
12. [x] 修改Decoder使用多层features
13. [x] 对比测试

---

## 📊 预期效果

### Phase 0完成后:
- 架构符合CrossMAE
- 但仍用self-attention
- 性能与原版相当

### Phase 1完成后:
- 完整CrossMAE
- 预计加速 3-4倍
- 性能相当或略好

### Phase 2完成后:
- CrossMAE + WeightedFeatureMaps
- 预计额外提升 0.1-0.3%
- 内存增加适中

---

这个更新后的计划是否清晰？主要新增了**Phase 0（架构调整）**，这是最关键的改动。
