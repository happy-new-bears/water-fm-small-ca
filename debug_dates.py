"""
调试脚本：检查parquet文件中的实际日期范围
"""

from datetime import datetime
from datasets.data_utils import load_vector_data_from_parquet

# 配置路径
vector_file = '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet'

print("=" * 60)
print("检查数据文件的日期范围")
print("=" * 60)

# 加载数据
vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
    vector_file,
    variables=['evaporation', 'discharge_vol'],
    start=datetime.strptime('1989-01-01', '%Y-%m-%d'),
    end=datetime.strptime('2015-12-31', '%Y-%m-%d'),
    nan_ratio=0.05,
)

print(f"\n数据加载完成:")
print(f"  形状: {vector_data.shape}")
print(f"  时间步数: {len(time_vec)}")
print(f"  流域数: {len(catchment_ids)}")

print(f"\n时间范围:")
print(f"  第一个日期: {time_vec[0]}")
print(f"  最后一个日期: {time_vec[-1]}")
print(f"  总天数: {len(time_vec)}")

# 检查特定日期
target_dates = [
    '1989-01-01',
    '2010-12-31',
    '2011-01-01',
    '2015-12-31',
]

print(f"\n检查目标日期是否存在:")
# Convert time_vec to datetime for comparison
time_vec_datetime = [datetime.strptime(str(d), '%Y-%m-%d') for d in time_vec]

for date_str in target_dates:
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    is_present = target_date in time_vec_datetime
    print(f"  {date_str}: {'✓ 存在' if is_present else '✗ 不存在'}")

    if not is_present:
        # 找到最接近的日期
        closest_idx = min(range(len(time_vec_datetime)), key=lambda i: abs((time_vec_datetime[i] - target_date).days))
        closest_date = time_vec_datetime[closest_idx]
        days_diff = (closest_date - target_date).days
        print(f"    最接近的日期: {closest_date.strftime('%Y-%m-%d')} (相差 {days_diff} 天)")

# 打印前10个和后10个日期
print(f"\n前10个日期:")
for i in range(min(10, len(time_vec))):
    print(f"  {i}: {time_vec[i]}")

print(f"\n后10个日期:")
for i in range(max(0, len(time_vec)-10), len(time_vec)):
    print(f"  {i}: {time_vec[i]}")

# 检查日期间隔
print(f"\n日期间隔检查:")
gaps = []
for i in range(1, min(100, len(time_vec))):
    gap = (time_vec[i] - time_vec[i-1]).days
    if gap != 1:
        gaps.append((i, time_vec[i-1], time_vec[i], gap))

if gaps:
    print(f"  发现 {len(gaps)} 个非连续日期:")
    for idx, date1, date2, gap in gaps[:10]:  # 只显示前10个
        print(f"    索引 {idx}: {date1} -> {date2} (间隔 {gap} 天)")
else:
    print(f"  ✓ 日期连续，无间隔")
