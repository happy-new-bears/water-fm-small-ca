"""
Analyze Catchment Spatial Distribution
Grouping 604 catchments into 10x10 spatial grid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style('whitegrid')
sns.set_palette('husl')

# Read catchment attribute data
data_root = '/Users/transformer/Desktop/water_data/new_version'
static_attr_file = f'{data_root}/Catchment_attributes/Catchment_attributes_nrfa.csv'

print("="*70)
print("Catchment Spatial Distribution Analysis")
print("="*70)

# Read data
df = pd.read_csv(static_attr_file)
print(f"\n✓ Loaded {len(df)} catchments")

# Check columns
print(f"\nAvailable columns: {list(df.columns[:20])}...")

# Extract lat/lon
if 'latitude' in df.columns and 'longitude' in df.columns:
    lats = df['latitude'].values
    lons = df['longitude'].values
    catchment_ids = df['id'].values
else:
    raise ValueError("latitude and longitude columns not found")

print(f"\nCoordinate ranges:")
print(f"  Latitude: {lats.min():.2f}° - {lats.max():.2f}°")
print(f"  Longitude: {lons.min():.2f}° - {lons.max():.2f}°")

# Create 10x10 grid
num_grid = 10
lat_bins = np.linspace(lats.min(), lats.max(), num_grid + 1)
lon_bins = np.linspace(lons.min(), lons.max(), num_grid + 1)

# Assign catchments to grid
lat_indices = np.digitize(lats, lat_bins) - 1
lon_indices = np.digitize(lons, lon_bins) - 1

# Handle boundary cases
lat_indices = np.clip(lat_indices, 0, num_grid - 1)
lon_indices = np.clip(lon_indices, 0, num_grid - 1)

# Calculate patch assignments
patch_assignments = lat_indices * num_grid + lon_indices  # [num_catchments]

# Count catchments per patch
patch_counts = np.zeros(num_grid * num_grid, dtype=int)
for patch_id in range(num_grid * num_grid):
    patch_counts[patch_id] = (patch_assignments == patch_id).sum()

# Reshape to 2D grid for visualization
patch_counts_2d = patch_counts.reshape(num_grid, num_grid)

print("\n" + "="*70)
print("Grouping Statistics")
print("="*70)

print(f"\nTotal patches: {num_grid}×{num_grid} = {num_grid * num_grid}")
print(f"Non-empty patches: {(patch_counts > 0).sum()}")
print(f"Empty patches: {(patch_counts == 0).sum()}")
print(f"\nCatchments per patch:")
print(f"  Min: {patch_counts.min()}")
print(f"  Max: {patch_counts.max()}")
print(f"  Mean: {patch_counts[patch_counts > 0].mean():.2f} (non-empty only)")
print(f"  Median: {np.median(patch_counts[patch_counts > 0]):.0f} (non-empty only)")
print(f"  Std: {patch_counts[patch_counts > 0].std():.2f}")

# Distribution stats
unique_counts = np.unique(patch_counts)
print(f"\nDistribution of catchments per patch:")
for count in sorted(unique_counts):
    num_patches = (patch_counts == count).sum()
    pct = num_patches / 100 * 100
    print(f"  {count:3d} catchments: {num_patches:3d} patches ({pct:.1f}%)")

# Create visualization
fig = plt.figure(figsize=(20, 6))

# 1. Heatmap - Catchments per patch
ax1 = plt.subplot(1, 3, 1)
im = ax1.imshow(patch_counts_2d, cmap='YlOrRd', origin='lower', aspect='auto')
ax1.set_title('Number of Catchments per Patch\n(10×10 Grid)',
              fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Longitude Bins', fontsize=13)
ax1.set_ylabel('Latitude Bins', fontsize=13)

# Add count annotations
for i in range(num_grid):
    for j in range(num_grid):
        count = patch_counts_2d[i, j]
        color = 'white' if count > patch_counts_2d.max() / 2 else 'black'
        ax1.text(j, i, str(count), ha='center', va='center',
                color=color, fontsize=11, fontweight='bold')

cbar1 = plt.colorbar(im, ax=ax1, label='Number of Catchments', pad=0.02)
cbar1.ax.tick_params(labelsize=11)

# 2. Histogram - Distribution
ax2 = plt.subplot(1, 3, 2)
bins = np.arange(0, patch_counts.max() + 2) - 0.5
n, bins, patches_hist = ax2.hist(patch_counts, bins=bins, color='steelblue',
                                  edgecolor='black', alpha=0.7, linewidth=1.5)

# Color the bars
for patch_bar, count in zip(patches_hist, n):
    if count > 0:
        patch_bar.set_facecolor(plt.cm.viridis(count / n.max()))

mean_val = patch_counts[patch_counts > 0].mean()
ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
           label=f'Mean = {mean_val:.1f}', zorder=5)
ax2.set_xlabel('Catchments per Patch', fontsize=13)
ax2.set_ylabel('Number of Patches', fontsize=13)
ax2.set_title('Distribution of Catchments per Patch',
              fontsize=16, fontweight='bold', pad=15)
ax2.legend(fontsize=12, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim(-1, patch_counts.max() + 1)

# 3. Scatter plot - Geographic distribution
ax3 = plt.subplot(1, 3, 3)

# Plot catchments colored by patch ID
scatter = ax3.scatter(lons, lats, c=patch_assignments, cmap='tab20',
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.7)

# Draw grid lines
for lat in lat_bins:
    ax3.axhline(lat, color='gray', linestyle='--', alpha=0.4, linewidth=1)
for lon in lon_bins:
    ax3.axvline(lon, color='gray', linestyle='--', alpha=0.4, linewidth=1)

ax3.set_xlabel('Longitude (°)', fontsize=13)
ax3.set_ylabel('Latitude (°)', fontsize=13)
ax3.set_title('Geographic Distribution of Catchments\n(Color = Patch ID)',
              fontsize=16, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.2, linestyle=':')

# Add colorbar
cbar3 = plt.colorbar(scatter, ax=ax3, label='Patch ID',
                     ticks=np.arange(0, 100, 10), pad=0.02)
cbar3.ax.tick_params(labelsize=11)

plt.tight_layout()

# Save figure
output_path = 'catchment_spatial_distribution_analysis_en.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Figure saved: {output_path}")

# Detailed analysis
print("\n" + "="*70)
print("Detailed Analysis")
print("="*70)

empty_patches = np.where(patch_counts == 0)[0]
if len(empty_patches) > 0:
    print(f"\n⚠️  Empty patches (total: {len(empty_patches)}):")
    for i, patch_id in enumerate(empty_patches[:10]):
        row = patch_id // num_grid
        col = patch_id % num_grid
        print(f"  Patch {patch_id:3d} (row {row}, col {col})")
    if len(empty_patches) > 10:
        print(f"  ... and {len(empty_patches) - 10} more")

crowded_patches = np.where(patch_counts >= np.percentile(patch_counts[patch_counts > 0], 90))[0]
if len(crowded_patches) > 0:
    print(f"\n📊 Crowded patches (top 10%, total: {len(crowded_patches)}):")
    for patch_id in crowded_patches[:10]:
        row = patch_id // num_grid
        col = patch_id % num_grid
        count = patch_counts[patch_id]
        print(f"  Patch {patch_id:3d} (row {row}, col {col}): {count} catchments")
    if len(crowded_patches) > 10:
        print(f"  ... and {len(crowded_patches) - 10} more")

# Save patch assignment data
output_data = {
    'patch_assignments': patch_assignments,
    'catchment_ids': catchment_ids,
    'num_patches': num_grid * num_grid,
    'patch_counts': patch_counts,
    'num_grid': num_grid,
    'lat_bins': lat_bins,
    'lon_bins': lon_bins,
}

output_file = 'data/catchment_spatial_patches_10x10.pt'
import torch
import os
os.makedirs('data', exist_ok=True)
torch.save(output_data, output_file)
print(f"\n✓ Patch assignment data saved: {output_file}")

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)

# Create a summary statistics table
print("\n" + "="*70)
print("Summary Table")
print("="*70)
summary_stats = {
    'Metric': [
        'Total Catchments',
        'Total Patches (10×10)',
        'Non-empty Patches',
        'Empty Patches',
        'Min Catchments/Patch',
        'Max Catchments/Patch',
        'Mean Catchments/Patch',
        'Median Catchments/Patch',
        'Std Catchments/Patch',
    ],
    'Value': [
        len(catchment_ids),
        num_grid * num_grid,
        (patch_counts > 0).sum(),
        (patch_counts == 0).sum(),
        patch_counts.min(),
        patch_counts.max(),
        f"{patch_counts[patch_counts > 0].mean():.2f}",
        f"{np.median(patch_counts[patch_counts > 0]):.0f}",
        f"{patch_counts[patch_counts > 0].std():.2f}",
    ]
}

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

plt.show()
