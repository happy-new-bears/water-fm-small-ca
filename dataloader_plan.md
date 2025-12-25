# DataLoader 完整实施计划

## 设计目标

构建一个支持**MAE风格多模态预训练**的DataLoader，具备以下特性：
1. **多尺度训练**：支持30-180天的可变序列长度
2. **随机mask预训练**：支持多段、不连续的时间段mask
3. **5模态数据**：3个图片模态 + 2个向量模态 + 静态属性
4. **灵活的下游适配**：Transformer天然支持任意长度输入

---
其他注意事项：
stride设置成config，不要固定，
maskratio你也设置成config 不要固定
## 一、数据格式说明

### 1.1 图片模态（3个）
- **precipitation**: `/Users/transformer/Desktop/water_data/new_version/precipitation_processed/`
- **soil_moisture**: `/Users/transformer/Desktop/water_data/new_version/soil_moisture_processed/`
- **temperature**: `/Users/transformer/Desktop/water_data/new_version/temperature_processed/`

**存储格式**：
- 每月一个h5文件，如 `2020-01.h5`, `2020-02.h5`
- h5文件内部：`{日期key: 290×180矩阵}`
- 日期key格式：需确认是 `'2020-01-01'` 还是 `'20200101'`（待确认）

### 1.2 向量模态（2个）
- **evaporation**: 604个catchment，每天一个值
- **riverflow**: 604个catchment，每天一个值
  - ⚠️ **注意**：1970-1988年数据缺失

**存储格式**（待确认）：
- 假设为 `.npy` 文件，形状 `[604, num_days]`
- 或其他格式（需要用户确认）

### 1.3 静态属性
- **文件路径**: `/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/`
- **推荐使用**: `Catchment_attributes_nrfa.csv`（包含所有属性）

**选用的14个属性**：
```python
STATIC_ATTRS = [
    # 基础地理 (2)
    "gauge_lat", "gauge_lon",

    # 地形特征 (7)
    "elev_min", "elev_max", "elev_mean",
    "elev_10", "elev_50", "elev_90",
    "area", "dpsbar",

    # 水体和人工影响 (3)
    "inwater_perc",
    "num_reservoir", "reservoir_cap",
]
```

### 1.4 时间范围
- **训练期**: 1989-01-01 至 2010-12-31（跳过riverflow缺失期）
- **验证期**: 2011-01-01 至 2015-12-31
- **测试期**: 2016-01-01 至 2020-12-31

---

## 二、整体架构设计

```
数据流程：

MultiModalHydroDataset (Dataset)
    ↓
返回固定长度序列（如180天完整数据）
    ↓
MultiScaleMaskedCollate (Collate Function)
    ↓
1. 随机采样序列长度 seq_len ∈ [30, 60, 90, 120, 180]
2. 截取每个样本的前seq_len天
3. 生成随机mask（多段、不连续）
    ↓
输出batch：
    - 数据: [B, seq_len, ...]（每个模态）
    - mask: [B, seq_len]（每个模态独立mask）
```

**关键设计原则**：
1. **Dataset简单化**：只负责加载数据，返回固定长度
2. **Collate灵活化**：负责长度采样、mask生成
3. **模态独立mask**：不同模态可以有不同的mask模式

---

## 三、核心类设计

### 3.1 MultiModalHydroDataset

**文件位置**: `water_fm/datasets/multimodal_dataset.py`

#### 初始化参数

```python
class MultiModalHydroDataset(Dataset):
    def __init__(
        self,
        # 图片模态路径
        precip_dir: str,
        soil_dir: str,
        temp_dir: str,

        # 向量模态数据（预加载）
        evap_data: np.ndarray,        # [604, num_days]
        riverflow_data: np.ndarray,   # [604, num_days]

        # 静态属性
        static_attr_file: str,
        static_attr_vars: List[str],

        # 时间范围
        start_date: str,              # '1989-01-01'
        end_date: str,                # '2020-12-31'

        # 采样参数
        max_sequence_length: int = 90,  # 返回的最大序列长度（内存限制）

        # Catchment配置
        catchment_ids: Optional[List[int]] = None,  # None表示使用全部604个

        # 其他
        split: str = 'train',  # 'train'/'val'/'test'
    )
```

#### 核心属性

```python
# 路径
self.precip_dir: Path
self.soil_dir: Path
self.temp_dir: Path

# 向量数据（内存中）
self.evap_data: np.ndarray        # [num_catchments, num_days]
self.riverflow_data: np.ndarray   # [num_catchments, num_days]

# 静态属性（内存中）
self.static_attrs: torch.Tensor   # [num_catchments, num_static_features]

# 时间索引
self.date_list: List[datetime]    # 所有日期
self.num_days: int

# Catchment
self.catchment_ids: np.ndarray    # 使用的catchment ID列表
self.num_catchments: int

# 采样参数
self.max_sequence_length: int = 90  # 内存限制

# h5文件映射（延迟加载）
self.h5_file_map: Dict[str, Dict[str, Path]]
# 格式: {'precip': {'2020-01': Path(...), ...}, 'soil': {...}, 'temp': {...}}

# 有效样本索引
self.valid_samples: List[Tuple[int, int]]
# [(catchment_idx, start_day_idx), ...]
```

#### 核心方法

**1. `__init__` - 初始化**

```python
def __init__(self, ...):
    # 保存参数
    self.precip_dir = Path(precip_dir)
    self.evap_data = evap_data
    ...

    # 生成日期列表
    self.date_list = self._generate_date_list(start_date, end_date)
    self.num_days = len(self.date_list)

    # 扫描h5文件
    self.h5_file_map = self._scan_h5_files()

    # 加载静态属性
    self.static_attrs = self._load_static_attributes(
        static_attr_file, catchment_ids, static_attr_vars
    )

    # 构建有效样本索引
    self.valid_samples = self._build_valid_samples()

    print(f"Dataset: {len(self.valid_samples)} valid samples, "
          f"{self.num_catchments} catchments, {self.num_days} days")
```

**2. `_generate_date_list` - 生成日期序列**

```python
def _generate_date_list(self, start_date: str, end_date: str) -> List[datetime]:
    """生成连续日期列表"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_list = []
    current = start
    while current <= end:
        date_list.append(current)
        current += timedelta(days=1)

    return date_list
```

**3. `_scan_h5_files` - 扫描h5文件**

```python
def _scan_h5_files(self) -> Dict[str, Dict[str, Path]]:
    """
    扫描三个图片模态的h5文件

    Returns:
        {
            'precip': {'2020-01': Path(...), '2020-02': Path(...), ...},
            'soil': {...},
            'temp': {...}
        }
    """
    file_map = {}

    for modality, dir_path in [
        ('precip', self.precip_dir),
        ('soil', self.soil_dir),
        ('temp', self.temp_dir)
    ]:
        file_map[modality] = {}

        # 扫描h5文件
        for h5_file in sorted(dir_path.glob('*.h5')):
            # 从文件名提取年月
            # 假设格式: 2020-01.h5 或 202001.h5
            stem = h5_file.stem
            if len(stem) == 7 and '-' in stem:  # 2020-01
                year_month = stem
            elif len(stem) == 6:  # 202001
                year_month = f"{stem[:4]}-{stem[4:6]}"
            else:
                print(f"Warning: Cannot parse {h5_file.name}")
                continue

            file_map[modality][year_month] = h5_file

    return file_map
```

**4. `_load_static_attributes` - 加载静态属性**

```python
def _load_static_attributes(
    self,
    file_path: str,
    catchment_ids: np.ndarray,
    vars_to_use: List[str]
) -> torch.Tensor:
    """
    从CSV加载静态属性并对齐

    Returns:
        torch.Tensor: [num_catchments, num_features]
    """
    import pandas as pd

    # 读取CSV
    df = pd.read_csv(file_path)

    # 去重
    df = df.drop_duplicates(subset=['id'], keep='first')

    # 筛选catchment
    df_filtered = df[df['id'].isin(catchment_ids)]

    # 检查缺失
    missing = set(catchment_ids) - set(df_filtered['id'])
    if missing:
        print(f"Warning: {len(missing)} catchments missing in attributes")

    # 按ID顺序排列
    df_ordered = df_filtered.set_index('id').loc[catchment_ids].reset_index()

    # 提取属性列
    try:
        attrs = df_ordered[vars_to_use].values.astype(np.float32)
    except KeyError as e:
        raise KeyError(f"Attributes {e} not found. Available: {list(df.columns)}")

    # 处理NaN
    if np.isnan(attrs).any():
        print("Warning: NaN in static attributes, filling with column means")
        col_means = np.nanmean(attrs, axis=0)
        for i in range(attrs.shape[1]):
            attrs[np.isnan(attrs[:, i]), i] = col_means[i]

    return torch.from_numpy(attrs)
```

**5. `_build_valid_samples` - 构建有效样本**

```python
def _build_valid_samples(self) -> List[Tuple[int, int]]:
    """
    构建有效样本索引

    规则：
    1. 必须有足够长度（max_sequence_length天）
    2. riverflow在窗口内不能有NaN
    """
    valid_samples = []

    for catch_idx in range(self.num_catchments):
        for day_idx in range(self.num_days - self.max_sequence_length + 1):
            # 提取窗口内的riverflow
            window_riverflow = self.riverflow_data[
                catch_idx,
                day_idx:day_idx + self.max_sequence_length
            ]

            # 检查是否有缺失
            if np.isnan(window_riverflow).any():
                continue

            # 通过检查
            valid_samples.append((catch_idx, day_idx))

    return valid_samples
```

**6. `__len__` - 返回样本数**

```python
def __len__(self) -> int:
    return len(self.valid_samples)
```

**7. `__getitem__` - 获取单个样本**

```python
def __getitem__(self, idx: int) -> Dict:
    """
    返回一个样本（max_sequence_length天的完整序列）

    Returns:
        {
            'precip': [T, 290, 180],
            'soil': [T, 290, 180],
            'temp': [T, 290, 180],
            'evap': [T],
            'riverflow': [T],
            'static_attr': [num_features],
            'catchment_idx': int,
            'catchment_id': int,
            'start_date': datetime,
        }
    """
    catchment_idx, start_day_idx = self.valid_samples[idx]
    end_day_idx = start_day_idx + self.max_sequence_length

    # 获取日期范围
    date_range = self.date_list[start_day_idx:end_day_idx]

    # 加载图片序列
    precip_seq = self._load_image_sequence('precip', date_range)
    soil_seq = self._load_image_sequence('soil', date_range)
    temp_seq = self._load_image_sequence('temp', date_range)

    # 获取向量序列
    evap_seq = self.evap_data[catchment_idx, start_day_idx:end_day_idx]
    riverflow_seq = self.riverflow_data[catchment_idx, start_day_idx:end_day_idx]

    return {
        'precip': precip_seq,           # [180, 290, 180]
        'soil': soil_seq,               # [180, 290, 180]
        'temp': temp_seq,               # [180, 290, 180]
        'evap': evap_seq,               # [180]
        'riverflow': riverflow_seq,     # [180]
        'static_attr': self.static_attrs[catchment_idx],  # [14]
        'catchment_idx': catchment_idx,
        'catchment_id': self.catchment_ids[catchment_idx],
        'start_date': date_range[0],
    }
```

**8. `_load_image_sequence` - 加载图片序列**

```python
def _load_image_sequence(
    self,
    modality: str,
    date_range: List[datetime]
) -> np.ndarray:
    """
    加载指定模态在date_range内的图片序列

    Args:
        modality: 'precip', 'soil', 或 'temp'
        date_range: 日期列表

    Returns:
        np.ndarray: [T, 290, 180]
    """
    T = len(date_range)
    result = np.zeros((T, 290, 180), dtype=np.float32)

    # 按月分组（优化：减少h5文件打开次数）
    from collections import defaultdict
    monthly_groups = defaultdict(list)
    for i, date in enumerate(date_range):
        year_month = date.strftime('%Y-%m')
        monthly_groups[year_month].append((i, date))

    # 逐月加载
    for year_month, date_indices in monthly_groups.items():
        h5_path = self.h5_file_map[modality][year_month]

        with h5py.File(h5_path, 'r') as f:
            for local_idx, date in date_indices:
                # 日期key格式（需要根据实际数据调整）
                date_key = date.strftime('%Y-%m-%d')  # 或 '%Y%m%d'

                try:
                    result[local_idx] = f[date_key][:]
                except KeyError:
                    # 尝试另一种格式
                    date_key_alt = date.strftime('%Y%m%d')
                    result[local_idx] = f[date_key_alt][:]

    return result
```

---

### 3.2 MultiScaleMaskedCollate

**文件位置**: `water_fm/datasets/collate.py`

#### 设计思路

在预训练阶段：
1. 每个batch随机选择一个序列长度（30/60/90/120/180）
2. 从每个样本的180天数据中截取前seq_len天
3. 为每个模态生成独立的随机mask
4. Mask策略：随机选择多个不连续的时间段

#### 实现

```python
class MultiScaleMaskedCollate:
    def __init__(
        self,
        # 序列长度选项
        seq_len_options: List[int] = [30, 60, 90, 120, 180],
        seq_len_probs: List[float] = [0.15, 0.20, 0.25, 0.20, 0.20],

        # Mask参数
        mask_ratio: float = 0.3,        # 总共mask多少比例的时间步
        num_mask_segments: int = 3,     # 分成几段mask

        # 模态mask策略
        mask_mode: str = 'mixed',  # 'independent', 'unified', 'mixed'

        # 模式
        mode: str = 'train',  # 'train', 'val', 'test'

        # 验证/测试时的固定长度
        val_seq_len: int = 90,
    ):
        self.seq_len_options = seq_len_options
        self.seq_len_probs = seq_len_probs
        self.mask_ratio = mask_ratio
        self.num_mask_segments = num_mask_segments
        self.mask_mode = mask_mode
        self.mode = mode
        self.val_seq_len = val_seq_len

        # 模态列表
        self.image_modalities = ['precip', 'soil', 'temp']
        self.vector_modalities = ['evap', 'riverflow']
        self.all_modalities = self.image_modalities + self.vector_modalities

    def __call__(self, batch_list: List[Dict]) -> Dict:
        """
        处理batch

        Args:
            batch_list: 来自Dataset的样本列表

        Returns:
            batch_dict: {
                # 数据
                'precip': [B, T, 290, 180],
                'soil': [B, T, 290, 180],
                'temp': [B, T, 290, 180],
                'evap': [B, T],
                'riverflow': [B, T],
                'static_attr': [B, num_features],

                # Mask（True=需要预测的位置）
                'precip_mask': [B, T],
                'soil_mask': [B, T],
                'temp_mask': [B, T],
                'evap_mask': [B, T],
                'riverflow_mask': [B, T],

                # 元信息
                'catchment_ids': [B],
                'seq_len': int,
            }
        """
        B = len(batch_list)

        # Step 1: 确定序列长度
        if self.mode == 'train':
            seq_len = np.random.choice(self.seq_len_options, p=self.seq_len_probs)
        else:
            seq_len = self.val_seq_len

        # Step 2: 截取数据
        truncated_batch = []
        for sample in batch_list:
            truncated = {
                key: val[:seq_len] if key in self.all_modalities else val
                for key, val in sample.items()
            }
            truncated_batch.append(truncated)

        # Step 3: 生成mask
        if self.mode == 'train':
            masks = self._generate_masks(B, seq_len)
        else:
            # 验证/测试时不mask
            masks = {mod: np.zeros((B, seq_len), dtype=bool)
                     for mod in self.all_modalities}

        # Step 4: Stack成batch
        batch_dict = {}

        # 图片模态
        for mod in self.image_modalities:
            batch_dict[mod] = torch.stack([
                torch.from_numpy(s[mod]) for s in truncated_batch
            ]).float()  # [B, T, 290, 180]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])

        # 向量模态
        for mod in self.vector_modalities:
            batch_dict[mod] = torch.stack([
                torch.from_numpy(s[mod]) for s in truncated_batch
            ]).float()  # [B, T]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])

        # 静态属性
        batch_dict['static_attr'] = torch.stack([
            s['static_attr'] for s in truncated_batch
        ])  # [B, 14]

        # 元信息
        batch_dict['catchment_ids'] = torch.tensor([
            s['catchment_id'] for s in truncated_batch
        ], dtype=torch.long)
        batch_dict['seq_len'] = seq_len

        return batch_dict

    def _generate_masks(self, B: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        为每个模态生成mask

        Args:
            B: batch size
            seq_len: 序列长度

        Returns:
            {modality: mask [B, seq_len]} for each modality
        """
        masks = {}

        if self.mask_mode == 'unified':
            # 所有模态使用相同的mask
            unified_mask = self._generate_single_batch_mask(B, seq_len)
            for mod in self.all_modalities:
                masks[mod] = unified_mask.copy()

        elif self.mask_mode == 'independent':
            # 每个模态独立生成mask
            for mod in self.all_modalities:
                masks[mod] = self._generate_single_batch_mask(B, seq_len)

        elif self.mask_mode == 'mixed':
            # 50%概率统一，50%概率独立
            if np.random.rand() < 0.5:
                unified_mask = self._generate_single_batch_mask(B, seq_len)
                for mod in self.all_modalities:
                    masks[mod] = unified_mask.copy()
            else:
                for mod in self.all_modalities:
                    masks[mod] = self._generate_single_batch_mask(B, seq_len)

        return masks

    def _generate_single_batch_mask(self, B: int, seq_len: int) -> np.ndarray:
        """
        为一个batch生成mask（每个样本可以不同）

        Args:
            B: batch size
            seq_len: 序列长度

        Returns:
            mask: [B, seq_len], True表示需要预测的位置
        """
        masks = []

        for _ in range(B):
            mask = self._generate_random_segments_mask(seq_len)
            masks.append(mask)

        return np.stack(masks, axis=0)

    def _generate_random_segments_mask(self, seq_len: int) -> np.ndarray:
        """
        生成随机多段mask

        Args:
            seq_len: 序列长度

        Returns:
            mask: [seq_len], True表示需要预测的位置
        """
        mask = np.zeros(seq_len, dtype=bool)

        total_to_mask = int(seq_len * self.mask_ratio)
        masked_count = 0

        # 随机生成num_mask_segments个mask段
        attempts = 0
        max_attempts = 20

        while masked_count < total_to_mask and attempts < max_attempts:
            # 随机起点
            start = np.random.randint(0, seq_len - 5)

            # 随机长度
            length = np.random.choice([5, 7, 10, 14, 21, 30])
            end = min(start + length, seq_len)

            # 设置mask
            mask[start:end] = True
            masked_count = mask.sum()
            attempts += 1

        return mask
```

---

## 四、使用示例

### 4.1 数据准备

首先需要准备向量模态数据：

```python
import numpy as np

# 假设你已经有evaporation和riverflow的数据
# 这里给出如何组织数据的示例

# 方案A：如果数据已经是numpy数组
evap_data = np.load('evaporation.npy')      # [604, num_days]
riverflow_data = np.load('riverflow.npy')  # [604, num_days]

# 方案B：如果需要从其他格式转换
# 例如从parquet文件读取
import polars as pl

df = pl.read_parquet('riverflow.parquet')
# 假设格式: columns=['date', 'ID', 'evaporation', 'riverflow']

# 转换为 [num_days, num_catchments, num_vars]
# 然后transpose为 [num_catchments, num_days]
# （具体实现参考现有代码的get_vectors函数）

# 标记riverflow的缺失期（1970-1988）
# 假设日期对应关系已知
for catch_idx in range(604):
    # 将1970-1988年的数据设为NaN
    riverflow_data[catch_idx, date_idx_1970:date_idx_1989] = np.nan
```

### 4.2 创建Dataset

```python
from datasets.multimodal_dataset import MultiModalHydroDataset

# 静态属性变量
STATIC_ATTRS = [
    "gauge_lat", "gauge_lon",
    "elev_min", "elev_max", "elev_mean",
    "elev_10", "elev_50", "elev_90",
    "area", "dpsbar",
    "inwater_perc",
    "num_reservoir", "reservoir_cap",
]

# 创建训练集
train_dataset = MultiModalHydroDataset(
    # 图片模态
    precip_dir='/Users/transformer/Desktop/water_data/new_version/precipitation_processed',
    soil_dir='/Users/transformer/Desktop/water_data/new_version/soil_moisture_processed',
    temp_dir='/Users/transformer/Desktop/water_data/new_version/temperature_processed',

    # 向量模态（预加载）
    evap_data=evap_data,
    riverflow_data=riverflow_data,

    # 静态属性
    static_attr_file='/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/Catchment_attributes_nrfa.csv',
    static_attr_vars=STATIC_ATTRS,

    # 时间范围
    start_date='1989-01-01',
    end_date='2010-12-31',

    # 序列长度
    max_sequence_length=180,

    # Catchment（默认使用全部604个）
    catchment_ids=None,

    split='train',
)

print(f"Training dataset: {len(train_dataset)} samples")
```

### 4.3 创建Collate Function

```python
from datasets.collate import MultiScaleMaskedCollate

# 预训练模式
train_collate = MultiScaleMaskedCollate(
    seq_len_options=[30, 60, 90, 120, 180],
    seq_len_probs=[0.15, 0.20, 0.25, 0.20, 0.20],
    mask_ratio=0.3,
    num_mask_segments=3,
    mask_mode='mixed',  # 混合mask策略
    mode='train',
)

# 验证模式
val_collate = MultiScaleMaskedCollate(
    seq_len_options=[90],  # 固定长度
    mask_ratio=0.0,        # 验证时不mask
    mode='val',
    val_seq_len=90,
)
```

### 4.4 创建DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=train_collate,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=val_collate,
    num_workers=4,
    pin_memory=True,
)
```

### 4.5 使用DataLoader

```python
for batch in train_loader:
    # 图片模态
    precip = batch['precip']          # [16, T, 290, 180] T每次不同
    soil = batch['soil']              # [16, T, 290, 180]
    temp = batch['temp']              # [16, T, 290, 180]

    # 向量模态
    evap = batch['evap']              # [16, T]
    riverflow = batch['riverflow']   # [16, T]

    # 静态属性
    static_attr = batch['static_attr']  # [16, 14]

    # Mask
    precip_mask = batch['precip_mask']      # [16, T] True=预测位置
    riverflow_mask = batch['riverflow_mask']  # [16, T]

    # 元信息
    seq_len = batch['seq_len']  # 当前batch的序列长度

    # 前向传播
    # outputs = model(batch)

    print(f"Batch: seq_len={seq_len}, "
          f"precip_mask_ratio={precip_mask.float().mean():.2f}")
```

---

## 五、文件组织结构

```
water_fm/
├── datasets/
│   ├── __init__.py
│   ├── multimodal_dataset.py    # MultiModalHydroDataset
│   └── collate.py               # MultiScaleMaskedCollate
├── utils/
│   └── data_utils.py            # 辅助函数（可选）
└── config.py                     # 配置参数
```

**创建文件：**
1. `water_fm/datasets/__init__.py`
2. `water_fm/datasets/multimodal_dataset.py`
3. `water_fm/datasets/collate.py`

---

## 六、下游任务适配

### 6.1 下游任务Dataset

对于下游任务（如预测未来7天），可以创建专门的Dataset：

```python
class DownstreamDataset(Dataset):
    def __init__(
        self,
        base_dataset: MultiModalHydroDataset,
        history_len: int = 30,
        pred_len: int = 7,
    ):
        self.base_dataset = base_dataset
        self.history_len = history_len
        self.pred_len = pred_len

    def __getitem__(self, idx):
        # 获取完整序列
        sample = self.base_dataset[idx]

        # 截取历史和未来
        return {
            'history': {
                'precip': sample['precip'][:self.history_len],
                'soil': sample['soil'][:self.history_len],
                # ...
            },
            'future': {
                'riverflow': sample['riverflow'][
                    self.history_len:self.history_len+self.pred_len
                ],
            },
            'static_attr': sample['static_attr'],
        }
```

### 6.2 Encoder支持任意长度

```python
class TemporalEncoder(nn.Module):
    def __init__(self, d_model=256, max_len=256):
        super().__init__()
        # 位置编码：支持最大256天
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        self.transformer = nn.TransformerEncoder(...)

    def forward(self, x):
        B, T, D = x.shape  # T可以是7, 30, 90, 180...任意值

        # 动态截取位置编码
        pos = self.pos_embed[:, :T, :]

        x = x + pos
        x = self.transformer(x)

        return x  # [B, T, D]
```

**关键**：Transformer天然支持任意长度，所以预训练在90天、下游使用30天，完全没问题！

---

## 七、数据归一化方案

### 7.1 为什么需要归一化？

不同模态和不同catchment的数据量纲差异很大：
- 降水：可能从0-100mm
- 温度：可能从-10到30°C
- 径流：不同catchment差异巨大（流域面积不同）

**归一化可以：**
1. 加速模型收敛
2. 避免某些特征主导训练
3. 提高模型泛化能力

### 7.2 图片模态归一化（特殊处理）

#### 陆地/海洋Mask

**Mask文件**: `/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt`

```python
mask = torch.load('gb_temp_valid_mask_290x180.pt')  # [290, 180]
# mask == 1: 陆地像素（10,110个）- 参与归一化
# mask == 0: 海洋像素（42,090个）- 不参与归一化
```

#### 为什么海洋不参与？

- 海洋区域的数据对陆地水文预测无意义
- 如果包含海洋，会拉偏统计量（海洋像素占80%）
- 只统计陆地区域更准确

#### 统计量计算

对每个图片模态（precipitation, soil_moisture, temperature）分别计算：

```python
def compute_image_stats(
    data_dir: str,
    land_mask: torch.Tensor,
    modality: str,
    num_samples: int = 1000
) -> Dict[str, torch.Tensor]:
    """
    计算图片模态在陆地区域的统计量

    Args:
        data_dir: h5文件目录
        land_mask: [H, W] 陆地mask（1=陆地，0=海洋）
        modality: 'precip', 'soil', 或 'temp'
        num_samples: 采样多少个时间步计算统计量

    Returns:
        {
            'mean': torch.Tensor [1],  # 陆地区域均值
            'std': torch.Tensor [1],   # 陆地区域标准差
        }
    """
    # 1. 随机采样num_samples个时间步的数据
    all_land_values = []

    for i in range(num_samples):
        # 随机选择一个日期，加载图片
        img = load_random_image(data_dir, modality)  # [290, 180]

        # 只提取陆地像素的值
        land_values = img[land_mask == 1]  # [10110]
        all_land_values.append(land_values)

    # 2. 合并所有采样
    all_land_values = torch.cat(all_land_values)  # [1000 * 10110]

    # 3. 计算均值和标准差
    mean = all_land_values.mean()
    std = all_land_values.std()

    return {'mean': mean, 'std': std}
```

#### 归一化应用

```python
# 在__getitem__中加载图片后
precip_seq = self._load_image_sequence('precip', date_range)  # [T, 290, 180]

# 归一化（只在陆地区域生效）
land_mask = self.land_mask  # [290, 180]
precip_normalized = precip_seq.clone()

# 扩展mask到时间维度
land_mask_expanded = land_mask.unsqueeze(0).expand(T, -1, -1)  # [T, 290, 180]

# 只归一化陆地像素
precip_normalized[land_mask_expanded == 1] = (
    (precip_normalized[land_mask_expanded == 1] - self.precip_mean) /
    (self.precip_std + 1e-8)
)
# 海洋像素保持原值（通常为0或NaN）
```

**关键设计：**
- 统计量：只基于陆地像素
- 应用：也只归一化陆地像素
- 海洋像素：保持原值或设为0（反正会被mask忽略）

### 7.3 向量模态归一化（Per-Catchment）

#### 为什么Per-Catchment？

不同catchment的径流量级差异巨大：
- 小流域：径流可能0-10 m³/s
- 大流域：径流可能0-1000 m³/s

如果用全局统计量，会导致小流域的信号被淹没。

#### 统计量计算

```python
def compute_vector_stats(
    vector_data: np.ndarray,  # [num_catchments, num_days, num_features]
    catchment_ids: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    计算每个catchment的统计量

    Args:
        vector_data: [604, num_days, 2]  # 2个特征：evap, riverflow
        catchment_ids: [604]

    Returns:
        {
            'mean': torch.Tensor [604, 2],  # 每个catchment每个特征的均值
            'std': torch.Tensor [604, 2],   # 每个catchment每个特征的标准差
        }
    """
    num_catchments, num_days, num_features = vector_data.shape

    means = np.zeros((num_catchments, num_features))
    stds = np.zeros((num_catchments, num_features))

    for catch_idx in range(num_catchments):
        for feat_idx in range(num_features):
            # 提取该catchment该特征的所有时间步数据
            data = vector_data[catch_idx, :, feat_idx]  # [num_days]

            # 过滤NaN（riverflow在1970-1988有缺失）
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                means[catch_idx, feat_idx] = valid_data.mean()
                stds[catch_idx, feat_idx] = valid_data.std()
            else:
                means[catch_idx, feat_idx] = 0.0
                stds[catch_idx, feat_idx] = 1.0

    return {
        'mean': torch.from_numpy(means).float(),  # [604, 2]
        'std': torch.from_numpy(stds).float(),    # [604, 2]
    }
```

#### 归一化应用

```python
# 在__getitem__中
evap_seq = self.evap_data[catchment_idx, start_idx:end_idx]  # [90]
riverflow_seq = self.riverflow_data[catchment_idx, start_idx:end_idx]  # [90]

# 归一化（使用该catchment的统计量）
evap_normalized = (evap_seq - self.evap_mean[catchment_idx]) / (
    self.evap_std[catchment_idx] + 1e-8
)
riverflow_normalized = (riverflow_seq - self.riverflow_mean[catchment_idx]) / (
    self.riverflow_std[catchment_idx] + 1e-8
)
```

### 7.4 静态属性归一化（全局）

静态属性（如海拔、面积、经纬度）使用全局统计量：

```python
def compute_static_stats(
    static_attrs: torch.Tensor,  # [num_catchments, num_features]
) -> Dict[str, torch.Tensor]:
    """
    计算静态属性的全局统计量

    Returns:
        {
            'mean': torch.Tensor [num_features],
            'std': torch.Tensor [num_features],
        }
    """
    return {
        'mean': static_attrs.mean(dim=0),  # [14]
        'std': static_attrs.std(dim=0),    # [14]
    }
```

### 7.5 统计量缓存机制

#### 文件组织

```
water_fm/
├── datasets/
│   └── normalization_stats.py   # 统计量计算脚本
└── cache/
    └── normalization_stats.pt    # 缓存的统计量
```

#### 统计量文件格式

```python
stats = {
    # 图片模态（3个）
    'precip_mean': torch.Tensor([...]),  # [1]
    'precip_std': torch.Tensor([...]),   # [1]
    'soil_mean': torch.Tensor([...]),    # [1]
    'soil_std': torch.Tensor([...]),     # [1]
    'temp_mean': torch.Tensor([...]),    # [1]
    'temp_std': torch.Tensor([...]),     # [1]

    # 向量模态（2个）
    'evap_mean': torch.Tensor([...]),    # [604]
    'evap_std': torch.Tensor([...]),     # [604]
    'riverflow_mean': torch.Tensor([...]),  # [604]
    'riverflow_std': torch.Tensor([...]),   # [604]

    # 静态属性
    'static_mean': torch.Tensor([...]),  # [14]
    'static_std': torch.Tensor([...]),   # [14]

    # 元信息
    'land_mask': torch.Tensor([...]),    # [290, 180] 陆地mask
    'num_samples_used': 1000,            # 计算图片统计量时用了多少样本
}

torch.save(stats, 'cache/normalization_stats.pt')
```

#### 使用流程

```python
class MultiModalHydroDataset(Dataset):
    def __init__(self, ..., stats_cache_path: Optional[str] = None):
        ...

        # 加载或计算统计量
        if stats_cache_path and os.path.exists(stats_cache_path):
            print(f"Loading normalization stats from {stats_cache_path}")
            self.stats = torch.load(stats_cache_path)
        else:
            print("Computing normalization stats (this may take a while)...")
            self.stats = self._compute_all_stats()

            # 保存统计量
            if stats_cache_path:
                torch.save(self.stats, stats_cache_path)
                print(f"Saved normalization stats to {stats_cache_path}")

        # 提取统计量
        self.precip_mean = self.stats['precip_mean']
        self.precip_std = self.stats['precip_std']
        self.evap_mean = self.stats['evap_mean']  # [604]
        self.evap_std = self.stats['evap_std']    # [604]
        ...

    def _compute_all_stats(self) -> Dict:
        """首次运行时计算所有统计量"""
        # 加载陆地mask
        land_mask = torch.load(
            '/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt'
        )

        stats = {'land_mask': land_mask}

        # 1. 计算图片模态统计量
        for modality in ['precip', 'soil', 'temp']:
            img_stats = compute_image_stats(
                getattr(self, f'{modality}_dir'),
                land_mask,
                modality,
                num_samples=1000
            )
            stats[f'{modality}_mean'] = img_stats['mean']
            stats[f'{modality}_std'] = img_stats['std']

        # 2. 计算向量模态统计量
        # 合并evap和riverflow
        vector_data = np.stack([
            self.evap_data,      # [604, num_days]
            self.riverflow_data  # [604, num_days]
        ], axis=-1)  # [604, num_days, 2]

        vec_stats = compute_vector_stats(vector_data, self.catchment_ids)
        stats['evap_mean'] = vec_stats['mean'][:, 0]      # [604]
        stats['evap_std'] = vec_stats['std'][:, 0]        # [604]
        stats['riverflow_mean'] = vec_stats['mean'][:, 1]  # [604]
        stats['riverflow_std'] = vec_stats['std'][:, 1]    # [604]

        # 3. 计算静态属性统计量
        static_stats = compute_static_stats(self.static_attrs)
        stats['static_mean'] = static_stats['mean']  # [14]
        stats['static_std'] = static_stats['std']    # [14]

        stats['num_samples_used'] = 1000

        return stats
```

### 7.6 归一化集成到Dataset

修改`__getitem__`方法，返回归一化后的数据：

```python
def __getitem__(self, idx: int) -> Dict:
    catchment_idx, start_day_idx = self.valid_samples[idx]
    end_day_idx = start_day_idx + self.max_sequence_length

    date_range = self.date_list[start_day_idx:end_day_idx]

    # 加载原始数据
    precip_seq = self._load_image_sequence('precip', date_range)
    soil_seq = self._load_image_sequence('soil', date_range)
    temp_seq = self._load_image_sequence('temp', date_range)
    evap_seq = self.evap_data[catchment_idx, start_day_idx:end_day_idx]
    riverflow_seq = self.riverflow_data[catchment_idx, start_day_idx:end_day_idx]

    # 归一化图片（只在陆地区域）
    precip_norm = self._normalize_image(precip_seq, 'precip')
    soil_norm = self._normalize_image(soil_seq, 'soil')
    temp_norm = self._normalize_image(temp_seq, 'temp')

    # 归一化向量（per-catchment）
    evap_norm = (evap_seq - self.stats['evap_mean'][catchment_idx]) / (
        self.stats['evap_std'][catchment_idx] + 1e-8
    )
    riverflow_norm = (riverflow_seq - self.stats['riverflow_mean'][catchment_idx]) / (
        self.stats['riverflow_std'][catchment_idx] + 1e-8
    )

    # 归一化静态属性（全局）
    static_norm = (
        self.static_attrs[catchment_idx] - self.stats['static_mean']
    ) / (self.stats['static_std'] + 1e-8)

    return {
        'precip': precip_norm,
        'soil': soil_norm,
        'temp': temp_norm,
        'evap': evap_norm,
        'riverflow': riverflow_norm,
        'static_attr': static_norm,
        'catchment_idx': catchment_idx,
        'catchment_id': self.catchment_ids[catchment_idx],
        'start_date': date_range[0],
    }

def _normalize_image(
    self,
    img_seq: np.ndarray,  # [T, 290, 180]
    modality: str
) -> np.ndarray:
    """归一化图片序列（只在陆地区域）"""
    img_norm = img_seq.copy()
    land_mask = self.stats['land_mask'].numpy()  # [290, 180]

    mean = self.stats[f'{modality}_mean'].item()
    std = self.stats[f'{modality}_std'].item()

    # 对每个时间步
    for t in range(img_seq.shape[0]):
        # 只归一化陆地像素
        img_norm[t][land_mask == 1] = (
            (img_seq[t][land_mask == 1] - mean) / (std + 1e-8)
        )
        # 海洋像素保持原值或设为0
        img_norm[t][land_mask == 0] = 0.0

    return img_norm
```

### 7.7 创建统计量计算脚本

**文件**: `water_fm/datasets/compute_normalization_stats.py`

这是一个独立脚本，用于预先计算并保存统计量：

```python
"""
预计算归一化统计量并保存

运行：
python datasets/compute_normalization_stats.py
"""

import torch
import numpy as np
from pathlib import Path
from multimodal_dataset import MultiModalHydroDataset

# 数据路径配置
DATA_CONFIG = {
    'precip_dir': '/Users/transformer/Desktop/water_data/new_version/precipitation_processed',
    'soil_dir': '/Users/transformer/Desktop/water_data/new_version/soil_moisture_processed',
    'temp_dir': '/Users/transformer/Desktop/water_data/new_version/temperature_processed',
    'evap_data': np.load('path/to/evaporation.npy'),
    'riverflow_data': np.load('path/to/riverflow.npy'),
    'static_attr_file': '/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/Catchment_attributes_nrfa.csv',
    'start_date': '1989-01-01',
    'end_date': '2010-12-31',
}

# 创建dataset（会自动计算统计量）
dataset = MultiModalHydroDataset(
    **DATA_CONFIG,
    stats_cache_path='cache/normalization_stats.pt',  # 会自动保存
)

print("Normalization stats computed and saved!")
print(f"Precip mean: {dataset.stats['precip_mean'].item():.4f}")
print(f"Precip std: {dataset.stats['precip_std'].item():.4f}")
```

### 7.8 使用示例

```python
# 训练时
train_dataset = MultiModalHydroDataset(
    ...,
    stats_cache_path='cache/normalization_stats.pt',  # 自动加载缓存
)

# 如果缓存不存在，会自动计算并保存
# 如果缓存存在，会直接加载（秒级）

# 验证集和测试集使用相同的统计量
val_dataset = MultiModalHydroDataset(
    ...,
    stats_cache_path='cache/normalization_stats.pt',  # 使用训练集的统计量
)
```

---

## 八、关键注意事项

### 8.1 需要用户确认的问题

1. **h5文件日期key格式**：
   - 是 `'2020-01-01'` 还是 `'20200101'`？
   - 需要打开一个h5文件查看

2. **向量数据格式**：
   - evaporation和riverflow的实际文件格式？
   - 如何加载和组织成 `[604, num_days]`？

3. **h5文件命名**：
   - 月份文件是 `2020-01.h5` 还是 `202001.h5`？

### 8.2 性能优化建议

1. **h5文件缓存**：
   - 可以添加LRU缓存，缓存最近打开的h5文件
   - 减少重复打开文件的开销

2. **多进程加载**：
   - 使用 `num_workers>0` 时要注意h5文件的线程安全
   - 建议在每个worker进程中独立打开h5文件

3. **内存优化**：
   - 图片数据按需加载（当前设计）
   - 向量数据较小，预加载到内存（当前设计）
   - 静态属性很小，预加载（当前设计）

### 7.3 数据验证

在开始训练前，建议进行以下验证：

```python
# 1. 验证单个样本
sample = train_dataset[0]
print("Sample keys:", sample.keys())
print("Precip shape:", sample['precip'].shape)  # 应该是 [180, 290, 180]
print("Riverflow shape:", sample['riverflow'].shape)  # 应该是 [180]

# 2. 验证batch
batch = next(iter(train_loader))
print("Batch keys:", batch.keys())
print("Precip batch shape:", batch['precip'].shape)  # [B, T, 290, 180]
print("Seq len:", batch['seq_len'])

# 3. 验证mask
print("Mask ratio:", batch['precip_mask'].float().mean())  # 应该接近0.3

# 4. 验证不同模态mask是否独立
print("Precip mask == Soil mask:",
      (batch['precip_mask'] == batch['soil_mask']).all())  # mixed模式可能True或False
```

---

## 八、总结

**DataLoader设计的核心优势**：

1. ✅ **简单清晰**：Dataset只负责加载，Collate负责采样和mask
2. ✅ **灵活性高**：支持多尺度训练和任意长度预测
3. ✅ **MAE风格**：随机多段mask，充分利用数据
4. ✅ **模态独立**：支持不同模态的独立mask策略
5. ✅ **下游友好**：预训练的encoder天然支持任意长度输入

**下一步**：确认数据格式细节后，即可开始实现代码！
