# 跨模态Multi-Modal MAE修改计划

**日期**: 2025-12-25
**目标**: 将独立模态MAE改造为跨模态融合MAE
**参考**: CAV-MAE + CrossMAE

---

## 📋 总体目标

将当前的**独立模态MAE**改造为**跨模态融合MAE**，实现：
1. 各模态添加modality tokens标识
2. Encoder输出通过shared transformer融合
3. Decoder接收融合后的multi-modal features进行cross-attention重建

---

## 🎯 修改任务分解

### **任务1: 添加Modality Tokens**

#### 1.1 在MultiModalMAE.__init__中定义10个modality tokens

**文件**: `models/multimodal_mae.py`
**位置**: 在创建encoders之前（约line 52之前）

**新增代码**:
```python
# ========== Modality Tokens (Encoder) ==========
# 5个encoder modality tokens (d_model维度 = 256)
self.modality_precip = nn.Parameter(torch.zeros(1, 1, config.d_model))
self.modality_soil = nn.Parameter(torch.zeros(1, 1, config.d_model))
self.modality_temp = nn.Parameter(torch.zeros(1, 1, config.d_model))
self.modality_evap = nn.Parameter(torch.zeros(1, 1, config.d_model))
self.modality_riverflow = nn.Parameter(torch.zeros(1, 1, config.d_model))

# ========== Decoder Modality Tokens ==========
# 5个decoder modality tokens (decoder_dim维度 = 128)
self.decoder_modality_precip = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
self.decoder_modality_soil = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
self.decoder_modality_temp = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
self.decoder_modality_evap = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
self.decoder_modality_riverflow = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
```

**初始化**: 在`__init__`末尾（创建所有模块后）添加
```python
def __init__(self, config, valid_patch_indices=None):
    super().__init__()

    # ... 现有的初始化代码 ...

    # ========== Initialize modality tokens ==========
    # Encoder modality tokens
    nn.init.normal_(self.modality_precip, std=0.02)
    nn.init.normal_(self.modality_soil, std=0.02)
    nn.init.normal_(self.modality_temp, std=0.02)
    nn.init.normal_(self.modality_evap, std=0.02)
    nn.init.normal_(self.modality_riverflow, std=0.02)

    # Decoder modality tokens
    nn.init.normal_(self.decoder_modality_precip, std=0.02)
    nn.init.normal_(self.decoder_modality_soil, std=0.02)
    nn.init.normal_(self.decoder_modality_temp, std=0.02)
    nn.init.normal_(self.decoder_modality_evap, std=0.02)
    nn.init.normal_(self.decoder_modality_riverflow, std=0.02)
```

**参考**: CAV-MAE line 88-89, 111-112, 153-156

---

#### 1.2 修改Image Encoder接收modality_token

**文件**: `models/image_encoder.py`

**修改1: __init__签名**（约line 38-50）
```python
def __init__(
    self,
    patch_size: int = 10,
    image_hw: Tuple[int, int] = (290, 180),
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    max_time_steps: int = 90,
    dropout: float = 0.1,
    valid_patch_indices: Tensor = None,
    use_weighted_fm: bool = False,
    use_fm_layers: list = None,
    use_input: bool = False,
    modality_token: nn.Parameter = None,  # ⭐ 新增参数
):
    super().__init__()

    # ... 现有初始化代码 ...

    # ⭐ 存储modality token引用
    self.modality_token = modality_token
```

**修改2: forward中添加modality token**（约line 213之后）

**当前代码**（line 212-213）:
```python
# Add both PEs to x (vectorized!)
x = x + spatial_emb + temporal_emb
```

**修改为**:
```python
# Add both PEs to x (vectorized!)
x = x + spatial_emb + temporal_emb

# ⭐ Add modality token (CAV-MAE style: after pos_embed)
if self.modality_token is not None:
    x = x + self.modality_token  # [1, 1, d_model] broadcast to [B, max_len, d_model]
```

**参考**: CAV-MAE line 275-276, 279-280

---

#### 1.3 修改Vector Encoder接收modality_token

**文件**: `models/vector_encoder.py`

**修改1: __init__签名**（约line 39-52）
```python
def __init__(
    self,
    in_feat: int = 1,
    stat_dim: int = 11,
    d_model: int = 256,
    n_layers: int = 4,
    nhead: int = 8,
    dropout: float = 0.1,
    max_len: int = 90,
    use_weighted_fm: bool = False,
    use_fm_layers: list = None,
    use_input: bool = False,
    patch_size: int = 8,
    modality_token: nn.Parameter = None,  # ⭐ 新增参数
):
    super().__init__()

    # ... 现有初始化代码 ...

    # ⭐ 存储modality token引用
    self.modality_token = modality_token
```

**修改2: forward中添加modality token**（约line 244之后）

**当前代码**（line 242-244）:
```python
# ===== Step 5: Flatten time dimension =====
# [B, max_len, T, d_model] -> [B, max_len*T, d_model]
x = x.reshape(B, max_len * T, self.d_model)
```

**修改为**:
```python
# ===== Step 5: Flatten time dimension =====
# [B, max_len, T, d_model] -> [B, max_len*T, d_model]
x = x.reshape(B, max_len * T, self.d_model)

# ⭐ Add modality token (CAV-MAE style: after pos_embed, before transformer)
if self.modality_token is not None:
    x = x + self.modality_token  # [1, 1, d_model] broadcast to [B, max_len*T, d_model]
```

**位置**: 在flatten之后、FiLM transformer之前

---

#### 1.4 修改MultiModalMAE创建encoder时传入modality_token

**文件**: `models/multimodal_mae.py`

**修改位置**: encoder创建代码（约line 53-122）

**当前代码**:
```python
self.precip_encoder = ImageModalityEncoder(
    patch_size=config.patch_size,
    image_hw=(config.image_height, config.image_width),
    d_model=config.d_model,
    nhead=config.nhead,
    num_layers=config.img_encoder_layers,
    max_time_steps=config.max_time_steps,
    dropout=config.dropout,
    valid_patch_indices=self.valid_patch_indices,
    use_weighted_fm=config.use_weighted_fm,
    use_fm_layers=config.use_fm_layers,
    use_input=config.use_input,
)
```

**修改为**:
```python
self.precip_encoder = ImageModalityEncoder(
    patch_size=config.patch_size,
    image_hw=(config.image_height, config.image_width),
    d_model=config.d_model,
    nhead=config.nhead,
    num_layers=config.img_encoder_layers,
    max_time_steps=config.max_time_steps,
    dropout=config.dropout,
    valid_patch_indices=self.valid_patch_indices,
    use_weighted_fm=config.use_weighted_fm,
    use_fm_layers=config.use_fm_layers,
    use_input=config.use_input,
    modality_token=self.modality_precip,  # ⭐ 新增
)
```

**类似修改所有5个encoder**:
- `self.precip_encoder` → 传入 `self.modality_precip`
- `self.soil_encoder` → 传入 `self.modality_soil`
- `self.temp_encoder` → 传入 `self.modality_temp`
- `self.evap_encoder` → 传入 `self.modality_evap`
- `self.riverflow_encoder` → 传入 `self.modality_riverflow`

---

### **任务2: 实现Shared Fusion Layers**

#### 2.1 在MultiModalMAE.__init__中添加shared transformer

**文件**: `models/multimodal_mae.py`
**位置**: 在decoders定义之前（约line 125之前）

**新增代码**:
```python
# ========== Shared Fusion Transformer ==========
# 参考CAV-MAE的blocks_u (unified branch)
# 让多个模态的visible tokens互相交互
self.shared_depth = getattr(config, 'shared_depth', 1)  # 默认1层

self.blocks_shared = nn.ModuleList([
    nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=4 * config.d_model,
        dropout=config.dropout,
        batch_first=True,
    )
    for _ in range(self.shared_depth)
])

# Normalization for fused features
self.norm_shared = nn.LayerNorm(config.d_model)
```

**参考**: CAV-MAE line 99, 302-304

---

#### 2.2 修改forward方法，添加fusion步骤

**文件**: `models/multimodal_mae.py`
**位置**: 在encoder调用之后、decoder调用之前（约line 220-240之间）

**当前代码**:
```python
# ===== Encode all modalities =====
# Image modalities
precip_token, precip_mask_info = self.precip_encoder(
    batch['precip'], batch['precip_mask']
)
soil_token, soil_mask_info = self.soil_encoder(
    batch['soil'], batch['soil_mask']
)
temp_token, temp_mask_info = self.temp_encoder(
    batch['temp'], batch['temp_mask']
)

# Vector modalities (with FiLM)
evap_token, evap_mask_info = self.evap_encoder(
    batch['evap'], batch['static_attr'], batch['evap_mask']
)
riverflow_token, riverflow_mask_info = self.riverflow_encoder(
    batch['riverflow'], batch['static_attr'], batch['riverflow_mask']
)

# ===== Decode all modalities =====
# ...
```

**插入fusion代码**（在encode和decode之间）:
```python
# ===== Encode all modalities ===== (保持不变)
# ... encoder调用 ...

# ===== Shared Fusion Layers ===== (⭐ 新增整个section)
# Step 1: 获取batch size和device
B = precip_token.shape[0]
device = precip_token.device

# Step 2: 拼接所有模态的visible tokens
# 注意: Vector modality的最后一个token是static token，需要排除
all_tokens = torch.cat([
    precip_token,              # [B, L_precip, d_model]
    soil_token,                # [B, L_soil, d_model]
    temp_token,                # [B, L_temp, d_model]
    evap_token[:, :-1, :],     # [B, L_evap-1, d_model] 排除static token
    riverflow_token[:, :-1, :] # [B, L_river-1, d_model] 排除static token
], dim=1)  # [B, L_total, d_model]

# Step 3: 创建padding mask (拼接各自的padding mask)
# 从mask_info中获取padding_mask，如果没有则创建全False的mask
precip_pad = precip_mask_info.get('padding_mask',
    torch.zeros(B, precip_token.shape[1], device=device, dtype=torch.bool))
soil_pad = soil_mask_info.get('padding_mask',
    torch.zeros(B, soil_token.shape[1], device=device, dtype=torch.bool))
temp_pad = temp_mask_info.get('padding_mask',
    torch.zeros(B, temp_token.shape[1], device=device, dtype=torch.bool))
evap_pad = evap_mask_info.get('padding_mask',
    torch.zeros(B, evap_token.shape[1], device=device, dtype=torch.bool))
riverflow_pad = riverflow_mask_info.get('padding_mask',
    torch.zeros(B, riverflow_token.shape[1], device=device, dtype=torch.bool))

# 排除vector modality的static token的padding (最后一个是static token)
evap_pad_seq = evap_pad[:, :-1] if evap_pad.shape[1] > 0 else evap_pad
riverflow_pad_seq = riverflow_pad[:, :-1] if riverflow_pad.shape[1] > 0 else riverflow_pad

all_padding_mask = torch.cat([
    precip_pad,
    soil_pad,
    temp_pad,
    evap_pad_seq,
    riverflow_pad_seq
], dim=1)  # [B, L_total]

# Step 4: 通过shared transformer进行跨模态融合
fused_features = all_tokens
for blk in self.blocks_shared:
    fused_features = blk(fused_features, src_key_padding_mask=all_padding_mask)
fused_features = self.norm_shared(fused_features)
# fused_features: [B, L_total, d_model] - 融合后的multi-modal features

# ===== Decode all modalities ===== (需要修改，见任务3)
# ...
```

**关键点**:
1. Vector modality要排除static token（最后一个token）
2. 正确拼接padding masks
3. Shared transformer接收`src_key_padding_mask`
4. 输出的`fused_features`包含所有模态的融合信息

**参考**: CAV-MAE line 299-304

---

### **任务3: 修改Decoder接收fused_features**

#### 3.1 修改Image Decoder的forward签名

**文件**: `models/image_decoder.py`

**修改1: forward签名**（约line 137）

**当前代码**:
```python
def forward(self, encoder_output, mask_info: Dict) -> Tensor:
```

**修改为**:
```python
def forward(self, encoder_output, mask_info: Dict, decoder_modality_token=None) -> Tensor:
```

**修改2: _forward_cross_attn签名和实现**（约line 149）

**当前代码**:
```python
def _forward_cross_attn(self, encoder_output, mask_info: Dict) -> Tensor:
```

**修改为**:
```python
def _forward_cross_attn(self, encoder_output, mask_info: Dict, decoder_modality_token=None) -> Tensor:
```

**修改3: 在queries创建后添加decoder_modality_token**（约line 202之后）

**当前代码**（line 191-202）:
```python
# Create Queries [B, k, decoder_dim]
queries = self.mask_token.expand(B, num_masked_per_sample, -1).clone()

# Add Spatial PE (Gathering, NO LOOP!)
spatial_emb = self.spatial_pos[0, p_indices]  # [B, k, decoder_dim]
queries = queries + spatial_emb

# Add Temporal PE (Gathering, NO LOOP!)
temporal_emb = self.temporal_pos.pe.squeeze(0)[t_indices]  # [B, k, decoder_dim]
queries = queries + temporal_emb
```

**修改为**:
```python
# Create Queries [B, k, decoder_dim]
queries = self.mask_token.expand(B, num_masked_per_sample, -1).clone()

# Add Spatial PE (Gathering, NO LOOP!)
spatial_emb = self.spatial_pos[0, p_indices]  # [B, k, decoder_dim]
queries = queries + spatial_emb

# Add Temporal PE (Gathering, NO LOOP!)
temporal_emb = self.temporal_pos.pe.squeeze(0)[t_indices]  # [B, k, decoder_dim]
queries = queries + temporal_emb

# ⭐ Add Decoder Modality Token (CAV-MAE style: after all PEs)
if decoder_modality_token is not None:
    queries = queries + decoder_modality_token  # [1, 1, decoder_dim] broadcast
```

**修改4: 传递decoder_modality_token**（约line 150）

**当前代码**:
```python
if self.use_cross_attn:
    return self._forward_cross_attn(encoder_output, mask_info)
```

**修改为**:
```python
if self.use_cross_attn:
    return self._forward_cross_attn(encoder_output, mask_info, decoder_modality_token)
```

**参考**: CAV-MAE line 338-339

---

#### 3.2 修改Vector Decoder的forward签名

**文件**: `models/vector_decoder.py`

**完全相同的修改步骤**（参考3.1）:

1. **forward签名**添加`decoder_modality_token=None`
2. **_forward_cross_attn签名**添加`decoder_modality_token=None`
3. **在queries创建后添加decoder_modality_token**（约line 195之后）:

```python
# Add Temporal PE (Gathering, NO LOOP!)
temporal_emb = self.temporal_pos.pe.squeeze(0)[t_indices]  # [B, k, decoder_dim]
queries = queries + temporal_emb

# ⭐ Add Decoder Modality Token (CAV-MAE style: after all PEs)
if decoder_modality_token is not None:
    queries = queries + decoder_modality_token  # [1, 1, decoder_dim] broadcast
```

4. **传递decoder_modality_token**到`_forward_cross_attn`

---

#### 3.3 修改MultiModalMAE的decoder调用

**文件**: `models/multimodal_mae.py`
**位置**: forward方法中的decoder调用部分（约line 240-248）

**当前代码**:
```python
# ===== Decode all modalities =====
# Image modalities
precip_pred = self.precip_decoder(precip_token, precip_mask_info)
soil_pred = self.soil_decoder(soil_token, soil_mask_info)
temp_pred = self.temp_decoder(temp_token, temp_mask_info)

# Vector modalities
evap_pred = self.evap_decoder(evap_token, evap_mask_info)
riverflow_pred = self.riverflow_decoder(riverflow_token, riverflow_mask_info)
```

**修改为**:
```python
# ===== Decode all modalities =====
# ⭐ 所有decoder现在接收fused_features（而非单模态token）

# Image modalities
precip_pred = self.precip_decoder(
    fused_features,                          # ⭐ 改为fused_features
    precip_mask_info,
    decoder_modality_token=self.decoder_modality_precip  # ⭐ 新增
)
soil_pred = self.soil_decoder(
    fused_features,                          # ⭐ 改为fused_features
    soil_mask_info,
    decoder_modality_token=self.decoder_modality_soil  # ⭐ 新增
)
temp_pred = self.temp_decoder(
    fused_features,                          # ⭐ 改为fused_features
    temp_mask_info,
    decoder_modality_token=self.decoder_modality_temp  # ⭐ 新增
)

# Vector modalities
evap_pred = self.evap_decoder(
    fused_features,                          # ⭐ 改为fused_features
    evap_mask_info,
    decoder_modality_token=self.decoder_modality_evap  # ⭐ 新增
)
riverflow_pred = self.riverflow_decoder(
    fused_features,                          # ⭐ 改为fused_features
    riverflow_mask_info,
    decoder_modality_token=self.decoder_modality_riverflow  # ⭐ 新增
)
```

**关键变化**:
1. 所有decoder接收`fused_features`（包含所有模态的融合信息）
2. 传入对应的`decoder_modality_token`（告诉decoder要重建哪个模态）

---

## 📊 修改总结表

| 任务 | 文件 | 修改类型 | 预计行数 |
|------|------|---------|---------|
| 1.1 定义10个modality tokens | `multimodal_mae.py` | 新增__init__ | ~25行 |
| 1.2 修改ImageEncoder | `image_encoder.py` | 修改__init__ + forward | ~8行 |
| 1.3 修改VectorEncoder | `vector_encoder.py` | 修改__init__ + forward | ~8行 |
| 1.4 创建encoder时传入token | `multimodal_mae.py` | 修改__init__ | ~5行×5 |
| 2.1 定义shared transformer | `multimodal_mae.py` | 新增__init__ | ~15行 |
| 2.2 实现fusion逻辑 | `multimodal_mae.py` | 修改forward | ~50行 |
| 3.1 修改ImageDecoder | `image_decoder.py` | 修改forward | ~10行 |
| 3.2 修改VectorDecoder | `vector_decoder.py` | 修改forward | ~10行 |
| 3.3 修改decoder调用 | `multimodal_mae.py` | 修改forward | ~20行 |

**总计**: 约156行代码修改

---

## ⚠️ 关键注意事项

### 1. **Vector Modality的Static Token处理**
```python
# ❌ 错误：包含static token
all_tokens = torch.cat([precip_token, ..., evap_token], dim=1)

# ✅ 正确：排除static token
all_tokens = torch.cat([precip_token, ..., evap_token[:, :-1, :]], dim=1)
```

### 2. **Modality Token添加顺序**
参考CAV-MAE (line 274-276):
```python
x = patch_embed(x)
x = x + pos_embed       # 先加位置编码
x = x + modality_token  # 再加模态标识
```

### 3. **Decoder Modality Token添加顺序**
参考CAV-MAE (line 336-339):
```python
queries = mask_token + spatial_PE + temporal_PE + decoder_modality_token
```
顺序：mask_token → spatial PE → temporal PE → modality token

### 4. **Padding Mask的正确拼接**
- 各模态的visible lengths不同
- Vector modality要排除static token的padding
- Shared transformer需要正确的`src_key_padding_mask`

### 5. **Config修改**
需要在`configs/mae_config.py`中添加:
```python
@dataclass
class MAEConfig:
    # ... 现有配置 ...

    # ⭐ 新增：Shared fusion transformer layers
    shared_depth: int = 1  # Number of shared fusion transformer layers
```

---

## 🔍 测试验证计划

修改完成后需要验证的内容：

### **1. 参数检查**
```python
import torch
from models.multimodal_mae import MultiModalMAE

model = MultiModalMAE(config)

# 检查modality tokens存在
assert hasattr(model, 'modality_precip')
assert hasattr(model, 'decoder_modality_precip')
# ... 检查其他8个

# 检查shared transformer存在
assert hasattr(model, 'blocks_shared')
assert len(model.blocks_shared) == config.shared_depth

print("✓ All parameters exist")
```

### **2. Forward Pass测试**
```python
# 创建测试数据
batch = {
    'precip': torch.randn(2, 90, 290, 180),
    'soil': torch.randn(2, 90, 290, 180),
    'temp': torch.randn(2, 90, 290, 180),
    'evap': torch.randn(2, 604, 90),
    'riverflow': torch.randn(2, 604, 90),
    'static_attr': torch.randn(2, 604, 11),
    'precip_mask': torch.rand(2, 90, 522) < 0.75,
    'soil_mask': torch.rand(2, 90, 522) < 0.75,
    'temp_mask': torch.rand(2, 90, 522) < 0.75,
    'evap_mask': torch.rand(2, 76, 90) < 0.75,
    'riverflow_mask': torch.rand(2, 76, 90) < 0.75,
}

# Forward pass
total_loss, loss_dict = model(batch)

print(f"✓ Total loss: {total_loss.item():.4f}")
print(f"✓ Loss dict: {loss_dict}")
```

### **3. 梯度检查**
```python
# Backward
total_loss.backward()

# 检查modality tokens有梯度
assert model.modality_precip.grad is not None
assert model.decoder_modality_precip.grad is not None
print("✓ Modality tokens have gradients")

# 检查shared transformer有梯度
for param in model.blocks_shared.parameters():
    assert param.grad is not None
print("✓ Shared transformer has gradients")
```

### **4. Shape检查**
```python
# 在forward中添加debug print
print(f"Fused features shape: {fused_features.shape}")
# 应该是 [B, L_total, d_model]

print(f"Precip pred shape: {precip_pred.shape}")
# 应该是 [B, T, num_patches, patch_dim]
```

---

## 📝 修改顺序建议

建议按以下顺序修改，每步验证后再进行下一步：

1. ✅ **第一步**: 添加modality tokens定义和初始化
   - 修改`multimodal_mae.py`的`__init__`
   - 测试：参数是否存在

2. ✅ **第二步**: 修改Encoder接收modality_token
   - 修改`image_encoder.py`和`vector_encoder.py`
   - 修改`multimodal_mae.py`创建encoder时传入token
   - 测试：forward是否正常

3. ✅ **第三步**: 添加shared transformer
   - 修改`multimodal_mae.py`的`__init__`
   - 测试：参数是否存在

4. ✅ **第四步**: 实现fusion逻辑
   - 修改`multimodal_mae.py`的`forward`
   - 测试：fusion后的shape是否正确

5. ✅ **第五步**: 修改Decoder签名
   - 修改`image_decoder.py`和`vector_decoder.py`
   - 支持`decoder_modality_token`参数（默认None，兼容旧代码）
   - 测试：forward是否正常

6. ✅ **第六步**: 修改decoder调用
   - 修改`multimodal_mae.py`的`forward`
   - 传入`fused_features`和`decoder_modality_token`
   - 测试：完整forward + backward

7. ✅ **第七步**: 完整测试
   - 运行training script
   - 检查loss是否下降
   - 检查wandb日志

---

## 💡 设计理念总结

### **为什么这样设计？**

1. **Encoder Modality Token**:
   - 让encoder知道自己处理的是什么模态
   - 在shared transformer中区分不同模态的tokens

2. **Shared Fusion Transformer**:
   - 让不同模态的visible tokens互相交互
   - 学习跨模态的依赖关系
   - 例如：降水和温度的联合模式

3. **Decoder Modality Token**:
   - 告诉decoder要重建哪个模态
   - 帮助decoder学习模态特定的重建策略
   - 例如：重建降水vs重建温度需要不同的策略

4. **Cross-Attention Decoder**:
   - Queries来自masked positions（要重建的部分）
   - Keys/Values来自fused_features（所有模态的融合信息）
   - 每个masked position可以从所有模态获取信息

### **与CAV-MAE和CrossMAE的对比**

| 维度 | CAV-MAE | CrossMAE | 我们的设计 |
|------|---------|----------|-----------|
| **Encoder** | 模态特定 + Shared | 单模态ViT | 模态特定 + Shared ✅ |
| **Modality Token** | ✅ 有 | ❌ 无 | ✅ 有（5个模态） |
| **Decoder输入** | All tokens (V+M) | Only masked | Only masked ✅ |
| **Decoder机制** | Self-Attention | Cross-Attention | Cross-Attention ✅ |
| **Keys/Values** | 各自模态 | 单模态visible | **所有模态fusion** ✅ |

**我们的创新点**：
- ✅ CAV-MAE的多模态encoder融合机制
- ✅ CrossMAE的高效cross-attention decoder
- ✅ **融合两者优点**：decoder的K/V来自multi-modal fused features

---

## 🎓 参考文献

1. **CAV-MAE**: Contrastive Audio-Visual Masked Autoencoder
   - 文件：`/Users/transformer/Desktop/water_code/cav-mae-master/src/models/cav_mae.py`
   - 关键设计：Modality tokens, Shared transformer layers

2. **CrossMAE**: Cross-Attention Masked Autoencoders
   - 文件：`/Users/transformer/Desktop/water_code/CrossMAE-main/models_cross.py`
   - 关键设计：Cross-attention decoder, Only masked queries

---

## ✅ 完成标准

修改完成后，需要满足以下所有条件：

- [ ] 10个modality tokens正确定义和初始化
- [ ] Encoder正确接收和添加modality tokens
- [ ] Shared transformer正确实现
- [ ] Fusion逻辑正确（排除vector static token）
- [ ] Decoder正确接收decoder_modality_token
- [ ] Decoder正确接收fused_features
- [ ] Forward pass成功
- [ ] Backward pass成功（所有参数有梯度）
- [ ] Loss能正常计算
- [ ] 完整训练能运行

---

**文档版本**: 1.0
**最后更新**: 2025-12-25
