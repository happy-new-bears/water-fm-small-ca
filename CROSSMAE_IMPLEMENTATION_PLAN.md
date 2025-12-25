# CrossMAE改造计划 - 水文多模态MAE

## 📋 总体目标
将基础MAE的Self-Attention Decoder改造为CrossMAE风格的Cross-Attention Decoder，
支持WeightedFeatureMaps（可选启用）

## 🎯 核心改动概览

### 需要修改的文件（按优先级）：
1. ✅ `models/layers.py` - 添加CrossAttention和CrossAttentionBlock
2. ✅ `models/image_encoder.py` - 支持输出多层features
3. ✅ `models/image_decoder.py` - 改造为CrossAttention decoder
4. ✅ `models/vector_encoder.py` - 支持输出多层features
5. ✅ `models/vector_decoder.py` - 改造为CrossAttention decoder
6. ✅ `configs/mae_config.py` - 添加CrossMAE相关配置
7. ✅ `models/multimodal_mae.py` - 适配新的encoder/decoder接口

---

## 📝 详细改动计划

### Step 1: 添加CrossAttention模块
**文件**: `models/layers.py`

**需要添加**:
```python
class CrossAttention(nn.Module):
    """
    Cross-Attention: queries from decoder, keys/values from encoder

    参考: CrossMAE transformer_utils.py:69-108
    """
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, ...):
        # Query projection (from decoder)
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        # Key-Value projection (from encoder)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        # ...

    def forward(self, x, y):
        """
        x: [B, N_masked, decoder_dim] - decoder queries
        y: [B, N_visible, encoder_dim] - encoder keys/values
        """
        # Q from x, K/V from y
        q = self.q(x)
        kv = self.kv(y)
        k, v = split kv

        # Cross-attention
        attn = (q @ k.T) * scale
        out = attn @ v
        return out

class CrossAttentionBlock(nn.Module):
    """
    Transformer block with cross-attention

    参考: CrossMAE transformer_utils.py:129-156
    """
    def __init__(self, encoder_dim, decoder_dim, num_heads,
                 self_attn=False, ...):
        # Optional self-attention (masked tokens之间)
        if self_attn:
            self.self_attn = Attention(decoder_dim, ...)

        # Cross-attention (masked query visible)
        self.cross_attn = CrossAttention(encoder_dim, decoder_dim, ...)

        # FFN
        self.mlp = Mlp(...)

    def forward(self, x, y):
        # Optional: masked self-attention
        if self.self_attn:
            x = x + self.self_attn(norm(x))

        # Cross-attention
        x = x + self.cross_attn(norm(x), y)

        # FFN
        x = x + self.mlp(norm(x))
        return x

class WeightedFeatureMaps(nn.Module):
    """
    学习如何组合多层encoder features

    参考: CrossMAE models_cross.py:23-40
    """
    def __init__(self, num_layers, embed_dim, decoder_depth):
        # 为每个decoder层学习encoder层的权重
        self.linear = nn.Linear(num_layers, decoder_depth, bias=False)
        # 初始化权重
        std = 1. / math.sqrt(num_layers)
        nn.init.normal_(self.linear.weight, mean=0., std=std)

    def forward(self, feature_maps):
        """
        feature_maps: list of [B, L, C] tensors
        Returns: [B, L, C, decoder_depth]
        """
        stacked = torch.stack(feature_maps, dim=-1)  # [B, L, C, k]
        output = self.linear(stacked)  # [B, L, C, decoder_depth]
        return output
```

---

### Step 2: 改造Image Encoder
**文件**: `models/image_encoder.py`

**关键改动**:

1. **添加配置参数** (在`__init__`):
```python
def __init__(self,
             ...,
             use_weighted_fm=False,  # 是否使用WeightedFeatureMaps
             use_fm_layers=None,     # 使用哪些层 [0, 2, 4, 5] or None (all)
             use_input=False):       # 是否包含输入作为第0层

    self.use_weighted_fm = use_weighted_fm
    self.use_input = use_input

    # 决定使用哪些层
    if use_fm_layers is None:
        self.use_fm_layers = list(range(num_layers))  # 所有层
    else:
        self.use_fm_layers = use_fm_layers
```

2. **修改forward函数** (参考 CrossMAE models_cross.py:205-230):
```python
def forward(self, x_img, patch_mask):
    """
    Returns:
        encoder_token: [B, d_model] or list of features
        mask_info: dict
    """
    B, T, H, W = x_img.shape

    # Patchify and filter valid patches
    patches = patchify(x_img, self.patch_size)
    patches = patches[:, :, self.valid_patch_indices, :]
    patch_mask_valid = patch_mask[:, :, self.valid_patch_indices]

    # Remove masked patches
    visible_patches_list = []
    visible_positions_list = []
    lengths = []

    for b in range(B):
        sample_patches = []
        sample_positions = []
        for t in range(T):
            visible_mask_t = ~patch_mask_valid[b, t]
            visible_patches_t = patches[b, t, visible_mask_t]
            sample_patches.append(visible_patches_t)
            # ... record positions
        visible_patches_list.append(torch.cat(sample_patches))
        lengths.append(len(sample_patches))

    # Pad to max_len
    max_len = max(lengths)
    x_padded = torch.zeros(B, max_len, self.patch_dim, device=x_img.device)
    for b in range(B):
        x_padded[b, :lengths[b]] = visible_patches_list[b]

    padding_mask = torch.zeros(B, max_len, device=x_img.device, dtype=torch.bool)
    for b in range(B):
        if lengths[b] < max_len:
            padding_mask[b, lengths[b]:] = True

    # Patch embedding
    x = self.patch_embed(x_padded)

    # Add position embeddings
    for b in range(B):
        for i, (t_idx, patch_idx) in enumerate(positions_padded[b]):
            x[b, i] += self.spatial_pos[0, patch_idx]
            x[b, i] += self.temporal_pos.pe[0, t_idx]

    # ===== 新增：收集多层features =====
    if self.use_weighted_fm:
        x_feats = []

        # 可选：添加输入作为第0层
        if self.use_input:
            x_feats.append(x.clone())

        # Transformer blocks
        for idx, blk in enumerate(self.transformer.layers):
            x = blk(x, src_key_padding_mask=padding_mask)

            # 保存指定层的输出
            if idx in self.use_fm_layers:
                x_feats.append(x.clone())

        # Pooling每一层
        encoder_tokens = []
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        for feat in x_feats:
            token = (feat * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
            encoder_tokens.append(self.norm(token))

        # 返回list of tokens
        mask_info = {
            'mask': patch_mask,
            'lengths': lengths,
            'padding_mask': padding_mask,
            'positions': positions_padded,
        }

        return encoder_tokens, mask_info  # list of [B, d_model]

    else:
        # 标准MAE：只返回最后一层
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pooling
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        encoder_token = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        encoder_token = self.norm(encoder_token)

        mask_info = {
            'mask': patch_mask,
            'lengths': lengths,
        }

        return encoder_token, mask_info  # [B, d_model]
```

---

### Step 3: 改造Image Decoder
**文件**: `models/image_decoder.py`

**重大改动** (参考 CrossMAE models_cross.py:240-256):

1. **修改初始化**:
```python
def __init__(self,
             encoder_dim: int = 256,
             decoder_dim: int = 128,
             num_patches: int = 522,
             patch_dim: int = 100,
             num_decoder_layers: int = 4,
             nhead: int = 8,
             max_time_steps: int = 90,
             dropout: float = 0.1,
             use_weighted_fm: bool = False,      # 新增
             num_encoder_layers: int = 6,        # 新增
             use_cross_attn: bool = True,        # 新增：是否用CrossAttention
             self_attn: bool = False):           # 新增：是否用masked self-attn

    super().__init__()

    self.encoder_dim = encoder_dim
    self.decoder_dim = decoder_dim
    self.use_weighted_fm = use_weighted_fm
    self.use_cross_attn = use_cross_attn

    # Project encoder token to decoder dimension (如果不用cross-attn)
    if not use_cross_attn:
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)

    # Mask token
    self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_dim))
    nn.init.normal_(self.mask_token, std=0.02)

    # WeightedFeatureMaps (可选)
    if use_weighted_fm:
        # 为每个decoder层准备norm
        self.dec_norms = nn.ModuleList([
            nn.LayerNorm(encoder_dim)
            for _ in range(num_decoder_layers)
        ])

        # Feature weighting module
        self.wfm = WeightedFeatureMaps(
            num_layers=num_encoder_layers,
            embed_dim=encoder_dim,
            decoder_depth=num_decoder_layers
        )

    # Decoder blocks
    if use_cross_attn:
        # CrossAttention blocks
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim,
                num_heads=nhead,
                self_attn=self_attn,  # 可选的masked self-attn
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])
    else:
        # 标准Self-Attention blocks (fallback)
        decoder_layer = nn.TransformerEncoderLayer(...)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

    # Spatial positional embedding
    self.spatial_pos = nn.Parameter(torch.zeros(1, 1, num_patches, decoder_dim))
    nn.init.normal_(self.spatial_pos, std=0.02)

    # Temporal positional encoding
    self.temporal_pos = PositionalEncoding(decoder_dim, max_time_steps)

    # Prediction head
    self.pred_head = nn.Linear(decoder_dim, patch_dim)
    self.decoder_norm = nn.LayerNorm(decoder_dim)
```

2. **关键：修改forward函数**:
```python
def forward(self, encoder_output, mask_info):
    """
    Args:
        encoder_output: [B, encoder_dim] or list of [B, encoder_dim]
        mask_info: dict with 'mask', 'lengths', etc.

    Returns:
        pred_patches: [B, T, num_patches, patch_dim]
    """
    mask = mask_info['mask']  # [B, T, num_patches]
    B, T, num_patches = mask.shape

    if self.use_cross_attn:
        # ===== CrossMAE风格 =====
        return self._forward_cross_attn(encoder_output, mask_info)
    else:
        # ===== 标准MAE风格 (fallback) =====
        return self._forward_self_attn(encoder_output, mask_info)

def _forward_cross_attn(self, encoder_output, mask_info):
    """CrossAttention版本的decoder"""
    mask = mask_info['mask']
    B, T, num_patches = mask.shape

    # Step 1: 只创建masked positions的queries
    # 参考 CrossMAE models_cross.py:232-238
    masked_queries = []
    masked_positions = []

    for b in range(B):
        for t in range(T):
            masked_indices = torch.where(mask[b, t])[0]  # True = masked
            num_masked_t = len(masked_indices)

            if num_masked_t > 0:
                # 为每个masked position创建query
                for patch_idx in masked_indices:
                    # mask_token + positional embedding
                    query = self.mask_token.squeeze() + \
                            self.spatial_pos[0, 0, patch_idx] + \
                            self.temporal_pos.pe[0, t]
                    masked_queries.append(query)
                    masked_positions.append((b, t, patch_idx.item()))

    if len(masked_queries) == 0:
        # Edge case: no masked patches
        return torch.zeros(B, T, num_patches, self.patch_dim, device=mask.device)

    # Stack all masked queries
    queries = torch.stack(masked_queries, dim=0)  # [total_masked, decoder_dim]
    queries = queries.unsqueeze(0)  # [1, total_masked, decoder_dim]

    # Step 2: Prepare encoder outputs as keys/values
    if self.use_weighted_fm:
        # encoder_output是list of [B, encoder_dim]
        # 需要组合成 [B, encoder_dim, decoder_depth]
        encoder_feats = self.wfm(encoder_output)  # [B, encoder_dim, decoder_depth]
    else:
        # encoder_output是单个 [B, encoder_dim]
        encoder_feats = encoder_output.unsqueeze(1)  # [B, 1, encoder_dim]

    # Step 3: CrossAttention decoder
    x = queries

    if self.use_weighted_fm:
        # 每个decoder层用不同的encoder feature组合
        for i, blk in enumerate(self.decoder_blocks):
            # 获取第i个decoder层对应的encoder features
            y = self.dec_norms[i](encoder_feats[..., i])  # [B, encoder_dim]
            y = y.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, total_masked, encoder_dim]

            x = blk(x, y)  # CrossAttentionBlock
    else:
        # 所有decoder层用相同的encoder output
        y = encoder_feats.expand(-1, x.shape[1], -1)  # [B, total_masked, encoder_dim]

        for blk in self.decoder_blocks:
            x = blk(x, y)

    # Step 4: Prediction
    x = self.decoder_norm(x)
    predictions = self.pred_head(x)  # [1, total_masked, patch_dim]

    # Step 5: 重组回 [B, T, num_patches, patch_dim]
    pred_patches = torch.zeros(B, T, num_patches, self.patch_dim,
                               device=mask.device, dtype=predictions.dtype)

    for idx, (b, t, patch_idx) in enumerate(masked_positions):
        pred_patches[b, t, patch_idx] = predictions[0, idx]

    return pred_patches

def _forward_self_attn(self, encoder_token, mask_info):
    """标准Self-Attention版本 (fallback)"""
    # 这是你现有的代码逻辑
    # ... 保持不变
    pass
```

---

### Step 4: Vector Encoder改造
**文件**: `models/vector_encoder.py`

**改动**: 与Image Encoder类似，支持输出多层features

```python
def __init__(self, ...,
             use_weighted_fm=False,
             use_fm_layers=None,
             use_input=False):
    # ... 同Image Encoder

def forward(self, x_vec, vec_mask, static_attr):
    # ... 处理visible vectors

    if self.use_weighted_fm:
        x_feats = []

        if self.use_input:
            x_feats.append(x.clone())

        for idx, blk in enumerate(self.transformer.layers):
            x = blk(x, src_key_padding_mask=padding_mask)

            if idx in self.use_fm_layers:
                x_feats.append(x.clone())

        # Pooling each layer
        encoder_tokens = []
        for feat in x_feats:
            token = ... # pooling
            encoder_tokens.append(token)

        return encoder_tokens, mask_info  # list
    else:
        # 标准版本
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        encoder_token = ... # pooling
        return encoder_token, mask_info
```

---

### Step 5: Vector Decoder改造
**文件**: `models/vector_decoder.py`

**改动**: 与Image Decoder类似

```python
def __init__(self, ...,
             use_weighted_fm=False,
             num_encoder_layers=4,
             use_cross_attn=True,
             self_attn=False):
    # ... 同Image Decoder

    if use_cross_attn:
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(encoder_dim, decoder_dim, ...)
            for _ in range(num_decoder_layers)
        ])

    if use_weighted_fm:
        self.wfm = WeightedFeatureMaps(...)
        self.dec_norms = nn.ModuleList([...])

def forward(self, encoder_output, mask_info):
    if self.use_cross_attn:
        return self._forward_cross_attn(encoder_output, mask_info)
    else:
        return self._forward_self_attn(encoder_output, mask_info)
```

---

### Step 6: 配置文件
**文件**: `configs/mae_config.py`

**添加新配置**:
```python
class MAEConfig:
    # ... 现有配置

    # ========== CrossMAE Configuration ==========
    # Decoder type
    use_cross_attn = True  # Use CrossAttention instead of Self-Attention

    # Weighted Feature Maps
    use_weighted_fm = False  # Enable WeightedFeatureMaps (多层encoder features)
    use_fm_layers = None  # Which encoder layers to use [0, 2, 4, 5] or None (all)
    use_input = False  # Include input as layer 0

    # Optional masked self-attention in decoder
    decoder_self_attn = False  # Add self-attention in decoder (default: False)
```

---

### Step 7: 主模型适配
**文件**: `models/multimodal_mae.py`

**修改forward逻辑**:
```python
def forward(self, batch):
    # ... 前置处理

    # Encoder
    precip_output, precip_mask_info = self.precip_encoder(
        batch['precip'], batch['precip_mask']
    )
    # precip_output可能是 [B, D] 或 list of [B, D]

    # Decoder
    precip_pred = self.precip_decoder(precip_output, precip_mask_info)

    # ... 计算loss
```

---

## 🔧 实现步骤建议

### Phase 1: 基础CrossAttention（必须）
1. ✅ 实现CrossAttention和CrossAttentionBlock in `models/layers.py`
2. ✅ 修改Image Decoder支持CrossAttention
3. ✅ 修改Vector Decoder支持CrossAttention
4. ✅ 添加config选项 `use_cross_attn = True`
5. ✅ 测试单个modality (image或vector)

### Phase 2: WeightedFeatureMaps（可选功能）
1. ✅ 实现WeightedFeatureMaps in `models/layers.py`
2. ✅ 修改Image Encoder输出多层features
3. ✅ 修改Vector Encoder输出多层features
4. ✅ 修改Decoder接收和使用多层features
5. ✅ 添加config选项 `use_weighted_fm = False` (默认关闭)
6. ✅ 测试对比 with/without WeightedFeatureMaps

### Phase 3: 可选优化
1. ✅ 添加masked self-attention选项
2. ✅ 添加部分重建机制 (kept_mask_ratio)
3. ✅ Flash Attention支持

---

## ⚠️ 注意事项

### 兼容性
- ✅ 保持向后兼容：`use_cross_attn=False` 时回退到标准MAE
- ✅ 分阶段实现：先CrossAttention，再WeightedFeatureMaps
- ✅ 充分测试每个阶段

### 内存管理
- WeightedFeatureMaps会增加encoder输出的存储
- 需要权衡：性能提升 vs 内存占用
- 建议先在小数据集测试

### 超参数
- `use_fm_layers`: 建议用 [0, 3, 5] 而不是所有层
- `decoder_self_attn`: 默认False，除非时序依赖很强

---

## ✅ 验证清单

- [ ] CrossAttention forward/backward正常
- [ ] 只用CrossAttention时loss下降
- [ ] WeightedFeatureMaps可选开关工作
- [ ] 开启WeightedFeatureMaps后训练正常
- [ ] 内存占用在可接受范围
- [ ] 训练速度提升（CrossAttention vs Self-Attention）
- [ ] 对比实验：标准MAE vs CrossMAE vs CrossMAE+WFM

---

## 📊 预期效果

### 计算效率
- 标准MAE: 22s/batch
- CrossMAE: 预计 6-8s/batch (3-4x加速)
- CrossMAE+WFM: 预计 7-9s/batch (轻微增加)

### 性能
- CrossMAE: 应该与标准MAE相当或略好
- CrossMAE+WFM: 预计提升 0.1-0.3% (小幅)

---

这个计划是否清晰？你想从哪个步骤开始实现？
