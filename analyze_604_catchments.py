"""
Analyze 604 Catchments Spatial Distribution
1. Filter to only 604 catchments from parquet file
2. Check if catchment IDs correlate with geographic location
3. Visualize with ID-based coloring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style('whitegrid')

print("="*70)
print("604 Catchments Spatial Distribution Analysis")
print("="*70)

# Step 1: Load 604 catchment IDs from parquet
data_root = '/Users/transformer/Desktop/water_data/new_version'
parquet_file = f'{data_root}/riverflow_evaporation_604catchments_1970_2015.parquet'

print(f"\nLoading catchment IDs from: {parquet_file}")
df_parquet = pd.read_parquet(parquet_file)
print(f"Parquet columns: {list(df_parquet.columns)}")

# Extract catchment IDs
if 'ID' in df_parquet.columns:
    catchment_ids_604 = df_parquet['ID'].unique()
elif 'station' in df_parquet.columns:
    catchment_ids_604 = df_parquet['station'].unique()
else:
    raise ValueError("'ID' or 'station' column not found in parquet file")

catchment_ids_604 = np.sort(catchment_ids_604)
print(f"\n✓ Found {len(catchment_ids_604)} catchments in parquet file")
print(f"  ID range: {catchment_ids_604.min()} - {catchment_ids_604.max()}")

# Step 2: Load static attributes
static_attr_file = f'{data_root}/Catchment_attributes/Catchment_attributes_nrfa.csv'
df_attr = pd.read_csv(static_attr_file)

# Filter to only 604 catchments
df_604 = df_attr[df_attr['id'].isin(catchment_ids_604)].copy()
print(f"\n✓ Matched {len(df_604)} catchments in attributes file")

# Sort by ID to maintain order
df_604 = df_604.sort_values('id').reset_index(drop=True)

# Extract coordinates
lats = df_604['latitude'].values
lons = df_604['longitude'].values
ids = df_604['id'].values

print(f"\nCoordinate ranges:")
print(f"  Latitude: {lats.min():.2f}° - {lats.max():.2f}°")
print(f"  Longitude: {lons.min():.2f}° - {lons.max():.2f}°")

# Step 3: Analyze correlation between ID and geographic location
print("\n" + "="*70)
print("Correlation Analysis: ID vs Geographic Location")
print("="*70)

# Correlation between ID and latitude
corr_lat, p_lat = spearmanr(ids, lats)
print(f"\nSpearman correlation (ID vs Latitude):")
print(f"  r = {corr_lat:.4f}, p-value = {p_lat:.4e}")

# Correlation between ID and longitude
corr_lon, p_lon = spearmanr(ids, lons)
print(f"\nSpearman correlation (ID vs Longitude):")
print(f"  r = {corr_lon:.4f}, p-value = {p_lon:.4e}")

if abs(corr_lat) > 0.3 or abs(corr_lon) > 0.3:
    print(f"\n⚠️  Moderate to strong correlation detected!")
    print(f"   Catchment IDs ARE related to geographic location.")
else:
    print(f"\n✓ Weak correlation detected.")
    print(f"  Catchment IDs are NOT strongly related to geographic location.")

# Step 4: Create 10x10 grid analysis
num_grid = 10
lat_bins = np.linspace(lats.min(), lats.max(), num_grid + 1)
lon_bins = np.linspace(lons.min(), lons.max(), num_grid + 1)

lat_indices = np.digitize(lats, lat_bins) - 1
lon_indices = np.digitize(lons, lon_bins) - 1

lat_indices = np.clip(lat_indices, 0, num_grid - 1)
lon_indices = np.clip(lon_indices, 0, num_grid - 1)

patch_assignments = lat_indices * num_grid + lon_indices

# Count catchments per patch
patch_counts = np.zeros(num_grid * num_grid, dtype=int)
for patch_id in range(num_grid * num_grid):
    patch_counts[patch_id] = (patch_assignments == patch_id).sum()

patch_counts_2d = patch_counts.reshape(num_grid, num_grid)

print("\n" + "="*70)
print("10×10 Grid Statistics (604 catchments)")
print("="*70)

print(f"\nTotal patches: {num_grid}×{num_grid} = {num_grid * num_grid}")
print(f"Non-empty patches: {(patch_counts > 0).sum()}")
print(f"Empty patches: {(patch_counts == 0).sum()}")
print(f"\nCatchments per patch:")
print(f"  Min: {patch_counts[patch_counts > 0].min()}")
print(f"  Max: {patch_counts.max()}")
print(f"  Mean: {patch_counts[patch_counts > 0].mean():.2f}")
print(f"  Median: {np.median(patch_counts[patch_counts > 0]):.0f}")
print(f"  Std: {patch_counts[patch_counts > 0].std():.2f}")

# Step 5: Create comprehensive visualization
fig = plt.figure(figsize=(24, 6))

# Plot 1: Geographic distribution colored by ID (dark to bright)
ax1 = plt.subplot(1, 4, 1)

# Normalize IDs to [0, 1] for coloring
id_normalized = (ids - ids.min()) / (ids.max() - ids.min())

scatter1 = ax1.scatter(lons, lats, c=id_normalized, cmap='viridis',
                       s=80, alpha=0.8, edgecolors='black', linewidth=0.8,
                       vmin=0, vmax=1)

ax1.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
ax1.set_title('Geographic Distribution\n(Color = Catchment ID, Dark→Bright)',
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle=':')

cbar1 = plt.colorbar(scatter1, ax=ax1, label='Catchment ID (normalized)', pad=0.02)
cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar1.set_ticklabels([
    f'{ids.min():.0f}',
    f'{ids.min() + 0.25*(ids.max()-ids.min()):.0f}',
    f'{ids.min() + 0.5*(ids.max()-ids.min()):.0f}',
    f'{ids.min() + 0.75*(ids.max()-ids.min()):.0f}',
    f'{ids.max():.0f}'
])

# Plot 2: Heatmap of catchments per patch
ax2 = plt.subplot(1, 4, 2)
im = ax2.imshow(patch_counts_2d, cmap='YlOrRd', origin='lower', aspect='auto')
ax2.set_title('Catchments per Patch\n(10×10 Grid)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Longitude Bins', fontsize=13, fontweight='bold')
ax2.set_ylabel('Latitude Bins', fontsize=13, fontweight='bold')

for i in range(num_grid):
    for j in range(num_grid):
        count = patch_counts_2d[i, j]
        if count > 0:
            color = 'white' if count > patch_counts_2d.max() / 2 else 'black'
            ax2.text(j, i, str(count), ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

cbar2 = plt.colorbar(im, ax=ax2, label='Catchments Count', pad=0.02)

# Plot 3: ID vs Latitude scatter
ax3 = plt.subplot(1, 4, 3)
ax3.scatter(ids, lats, c=id_normalized, cmap='viridis',
           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Catchment ID', fontsize=13, fontweight='bold')
ax3.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
ax3.set_title(f'ID vs Latitude\n(Spearman r={corr_lat:.3f})',
              fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, linestyle=':')

# Add trend line
z = np.polyfit(ids, lats, 1)
p = np.poly1d(z)
ax3.plot(ids, p(ids), "r--", linewidth=2, alpha=0.7, label='Linear fit')
ax3.legend(fontsize=11)

# Plot 4: ID vs Longitude scatter
ax4 = plt.subplot(1, 4, 4)
ax4.scatter(ids, lons, c=id_normalized, cmap='viridis',
           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Catchment ID', fontsize=13, fontweight='bold')
ax4.set_ylabel('Longitude (°)', fontsize=13, fontweight='bold')
ax4.set_title(f'ID vs Longitude\n(Spearman r={corr_lon:.3f})',
              fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, linestyle=':')

# Add trend line
z = np.polyfit(ids, lons, 1)
p = np.poly1d(z)
ax4.plot(ids, p(ids), "r--", linewidth=2, alpha=0.7, label='Linear fit')
ax4.legend(fontsize=11)

plt.tight_layout()

# Save figure
output_path = 'catchment_604_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Figure saved: {output_path}")

# Save patch data for 604 catchments
output_data = {
    'catchment_ids': ids,
    'patch_assignments': patch_assignments,
    'num_patches': num_grid * num_grid,
    'patch_counts': patch_counts,
    'num_grid': num_grid,
    'lat_bins': lat_bins,
    'lon_bins': lon_bins,
    'num_catchments': len(ids),
}

import torch
import os
os.makedirs('data', exist_ok=True)
output_file = 'data/catchment_spatial_patches_604_10x10.pt'
torch.save(output_data, output_file)
print(f"✓ Patch data saved: {output_file}")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = f"""
Catchments: {len(ids)}
ID range: {ids.min()} - {ids.max()}

Spatial extent:
  Latitude: {lats.min():.2f}° - {lats.max():.2f}°
  Longitude: {lons.min():.2f}° - {lons.max():.2f}°

10×10 Grid:
  Non-empty patches: {(patch_counts > 0).sum()}/100
  Catchments/patch: {patch_counts[patch_counts > 0].mean():.1f} ± {patch_counts[patch_counts > 0].std():.1f}

ID-Location Correlation:
  ID vs Latitude: r = {corr_lat:.3f} (p={p_lat:.2e})
  ID vs Longitude: r = {corr_lon:.3f} (p={p_lon:.2e})

Conclusion: {"IDs CORRELATE with location" if abs(corr_lat) > 0.3 or abs(corr_lon) > 0.3 else "IDs do NOT strongly correlate with location"}
"""

print(summary)

print("="*70)
print("Analysis Complete!")
print("="*70)

plt.show()
