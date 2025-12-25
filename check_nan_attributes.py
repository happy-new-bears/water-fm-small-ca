"""
检查604个catchment的静态属性中的NaN情况
"""

import pandas as pd
import polars as pl
from datetime import datetime
import sys
sys.path.insert(0, '.')

from datasets.data_utils import load_vector_data_from_parquet

# 1. 加载604个catchment的ID
print("=" * 60)
print("加载604个catchment ID...")
print("=" * 60)

vector_file = '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet'
vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
    vector_file,
    variables=['evaporation', 'discharge_vol'],
    start=datetime.strptime('1989-01-01', '%Y-%m-%d'),
    end=datetime.strptime('2015-12-30', '%Y-%m-%d'),
    nan_ratio=0.05,
)

print(f"\n流域数量: {len(catchment_ids)}")
print(f"流域ID范围: {catchment_ids.min()} - {catchment_ids.max()}")
print(f"流域ID示例: {catchment_ids[:10]}")

# 2. 加载静态属性文件
print(f"\n{'=' * 60}")
print("加载静态属性文件...")
print("=" * 60)

static_attr_file = '/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/Catchment_attributes_nrfa.csv'
df_static = pd.read_csv(static_attr_file)

print(f"静态属性文件总共有 {len(df_static)} 个catchment")
print(f"静态属性列: {list(df_static.columns)}")

# 3. 筛选出604个catchment的数据
print(f"\n{'=' * 60}")
print("筛选604个catchment的静态属性...")
print("=" * 60)

# 检查ID列名
id_col = None
for col in ['ID', 'id', 'catchment_id', 'station_id', 'gauge_id']:
    if col in df_static.columns:
        id_col = col
        break

if id_col is None:
    print(f"警告: 找不到ID列，可用列: {list(df_static.columns)}")
    id_col = df_static.columns[0]
    print(f"使用第一列作为ID: {id_col}")

df_604 = df_static[df_static[id_col].isin(catchment_ids)]
print(f"匹配到 {len(df_604)} 个catchment")

if len(df_604) < len(catchment_ids):
    missing_ids = set(catchment_ids) - set(df_604[id_col])
    print(f"警告: {len(missing_ids)} 个catchment在静态属性文件中找不到")
    print(f"缺失的ID示例: {list(missing_ids)[:10]}")

# 4. 检查配置中使用的属性
print(f"\n{'=' * 60}")
print("检查配置中使用的静态属性...")
print("=" * 60)

static_attrs = [
    "latitude", "longitude",
    "minimum-altitude", "maximum-altitude", "50-percentile-altitude",
    "10-percentile-altitude", "90-percentile-altitude",
    "catchment-area", "dpsbar",
    "propwet", "bfihost",
]

print(f"配置中使用的属性: {static_attrs}")

# 检查哪些属性存在
available_attrs = [attr for attr in static_attrs if attr in df_604.columns]
missing_attrs = [attr for attr in static_attrs if attr not in df_604.columns]

print(f"\n可用的属性 ({len(available_attrs)}): {available_attrs}")
if missing_attrs:
    print(f"缺失的属性 ({len(missing_attrs)}): {missing_attrs}")

# 5. 详细检查NaN情况
print(f"\n{'=' * 60}")
print("检查604个catchment的NaN情况")
print("=" * 60)

for attr in available_attrs:
    nan_count = df_604[attr].isna().sum()
    nan_percentage = (nan_count / len(df_604)) * 100

    if nan_count > 0:
        print(f"\n⚠️  {attr}:")
        print(f"    NaN数量: {nan_count}/{len(df_604)} ({nan_percentage:.2f}%)")
        print(f"    有效值范围: [{df_604[attr].min():.4f}, {df_604[attr].max():.4f}]")
        print(f"    均值: {df_604[attr].mean():.4f}")

        # 显示包含NaN的catchment ID
        nan_catchments = df_604[df_604[attr].isna()][id_col].values
        print(f"    包含NaN的catchment ID: {nan_catchments[:10]}")
        if len(nan_catchments) > 10:
            print(f"    ... 还有 {len(nan_catchments) - 10} 个")
    else:
        print(f"✓ {attr}: 无NaN值")

# 6. 总结
print(f"\n{'=' * 60}")
print("总结")
print("=" * 60)

total_nan_count = 0
attrs_with_nan = []
for attr in available_attrs:
    nan_count = df_604[attr].isna().sum()
    total_nan_count += nan_count
    if nan_count > 0:
        attrs_with_nan.append(attr)

print(f"总NaN数量: {total_nan_count}")
print(f"包含NaN的属性数量: {len(attrs_with_nan)}/{len(available_attrs)}")
if attrs_with_nan:
    print(f"包含NaN的属性: {attrs_with_nan}")

# 7. 检查有多少个catchment至少有一个NaN
catchments_with_nan = df_604[available_attrs].isna().any(axis=1).sum()
print(f"\n至少有一个NaN的catchment数量: {catchments_with_nan}/{len(df_604)} ({catchments_with_nan/len(df_604)*100:.2f}%)")

# 8. 保存详细报告
print(f"\n{'=' * 60}")
print("保存详细报告...")
print("=" * 60)

# 创建报告
report = []
for idx, row in df_604.iterrows():
    catchment_id = row[id_col]
    nan_attrs = []
    for attr in available_attrs:
        if pd.isna(row[attr]):
            nan_attrs.append(attr)

    if nan_attrs:
        report.append({
            'catchment_id': catchment_id,
            'nan_count': len(nan_attrs),
            'nan_attributes': ', '.join(nan_attrs)
        })

if report:
    df_report = pd.DataFrame(report)
    df_report = df_report.sort_values('nan_count', ascending=False)

    output_file = 'nan_attributes_report.csv'
    df_report.to_csv(output_file, index=False)
    print(f"详细报告已保存到: {output_file}")

    print(f"\nNaN最多的前10个catchment:")
    print(df_report.head(10).to_string(index=False))
else:
    print("✓ 所有catchment的所有属性都没有NaN值！")
