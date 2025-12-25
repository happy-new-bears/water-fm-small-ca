"""
Visualize patch-level masking for both image and vector modalities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("Visualizing Patch-Level Masking")
print("="*70)

# ========== Image Modality ==========
# Image parameters
img_height, img_width = 290, 180
patch_size_img = 10
num_patches_h = img_height // patch_size_img  # 29
num_patches_w = img_width // patch_size_img   # 18
num_patches_img = num_patches_h * num_patches_w  # 522

# Load land mask to get valid patches
land_mask_path = '/Users/transformer/Desktop/water_data/new_version/gb_temp_valid_mask_290x180.pt'
land_mask = torch.load(land_mask_path).numpy()  # [290, 180]

# Calculate valid patches (land patches only)
valid_patch_indices = []
patch_idx = 0
for i in range(num_patches_h):
    for j in range(num_patches_w):
        patch = land_mask[
            i*patch_size_img:(i+1)*patch_size_img,
            j*patch_size_img:(j+1)*patch_size_img
        ]
        land_ratio = patch.sum() / (patch_size_img * patch_size_img)
        if land_ratio >= 0.5:  # Same threshold as collate
            valid_patch_indices.append(patch_idx)
        patch_idx += 1

valid_patch_indices = np.array(valid_patch_indices)
num_valid_patches = len(valid_patch_indices)

# Generate image mask for one timestep [num_patches]
# IMPORTANT: Only mask from valid patches!
mask_ratio = 0.6  # 60% mask ratio for visualization
num_to_mask_img = int(num_valid_patches * mask_ratio)
img_patch_mask = np.zeros(num_patches_img, dtype=bool)

# Randomly select which valid patches to mask
masked_valid_indices = np.random.choice(num_valid_patches, size=num_to_mask_img, replace=False)
masked_patch_indices = valid_patch_indices[masked_valid_indices]
img_patch_mask[masked_patch_indices] = True

# Reshape to 2D grid [29, 18]
img_patch_mask_2d = img_patch_mask.reshape(num_patches_h, num_patches_w)

# Create land validity mask for visualization
valid_mask_2d = np.zeros((num_patches_h, num_patches_w), dtype=bool)
for idx in valid_patch_indices:
    i = idx // num_patches_w
    j = idx % num_patches_w
    valid_mask_2d[i, j] = True

print(f"\n1. Image Modality:")
print(f"   Image size: {img_height}×{img_width}")
print(f"   Patch size: {patch_size_img}×{patch_size_img}")
print(f"   Patch grid: {num_patches_h}×{num_patches_w} = {num_patches_img} patches")
print(f"   Valid patches (land): {num_valid_patches}/{num_patches_img} ({num_valid_patches/num_patches_img:.1%})")
print(f"   Masked patches: {img_patch_mask.sum()}/{num_valid_patches} ({img_patch_mask.sum()/num_valid_patches:.1%})")

# ========== Vector Modality ==========
# Vector parameters
num_catchments = 604
patch_size_vec = 8
num_patches_vec = (num_catchments + patch_size_vec - 1) // patch_size_vec  # 76
num_padded = num_patches_vec * patch_size_vec  # 608

# Generate vector mask for one timestep [num_patches]
num_to_mask_vec = int(num_patches_vec * mask_ratio)
vec_patch_mask = np.zeros(num_patches_vec, dtype=bool)
masked_indices_vec = np.random.choice(num_patches_vec, size=num_to_mask_vec, replace=False)
vec_patch_mask[masked_indices_vec] = True

# Load catchment locations for visualization
import pandas as pd
parquet_file = '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet'
attr_file = '/Users/transformer/Desktop/water_data/Catchment_attributes/Catchment_attributes_nrfa.csv'

df_parquet = pd.read_parquet(parquet_file)
catchment_ids = df_parquet['ID'].unique()
catchment_ids = np.sort(catchment_ids)

df_attr = pd.read_csv(attr_file)
df_604 = df_attr[df_attr['id'].isin(catchment_ids)].copy()
lats = df_604['latitude'].values
lons = df_604['longitude'].values

print(f"\n2. Vector Modality:")
print(f"   Catchments: {num_catchments}")
print(f"   Patch size: {patch_size_vec} catchments/patch")
print(f"   Number of patches: {num_patches_vec}")
print(f"   Padded catchments: {num_padded} (padding: {num_padded - num_catchments})")
print(f"   Masked patches: {vec_patch_mask.sum()}/{num_patches_vec} ({vec_patch_mask.mean():.1%})")

# ========== Visualization ==========
fig = plt.figure(figsize=(20, 10))

# Create grid: 2 rows, 3 columns
# Row 1: Image modality (2 subplots)
# Row 2: Vector modality (3 subplots)

# ===== Row 1: Image Modality =====
# Subplot 1: Image patch grid with mask
ax1 = plt.subplot(2, 3, 1)
# Create 3-state visualization: 0=invalid (gray), 1=visible (blue), 2=masked (red)
patch_state = np.zeros((num_patches_h, num_patches_w))
for i in range(num_patches_h):
    for j in range(num_patches_w):
        idx = i * num_patches_w + j
        if not valid_mask_2d[i, j]:
            patch_state[i, j] = 0  # Invalid (ocean)
        elif img_patch_mask_2d[i, j]:
            patch_state[i, j] = 2  # Masked (land, masked)
        else:
            patch_state[i, j] = 1  # Visible (land, visible)

cmap_img = ListedColormap(['lightgray', 'lightblue', 'coral'])
im1 = ax1.imshow(patch_state, cmap=cmap_img, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
ax1.set_xlabel('Patch Column Index', fontsize=11)
ax1.set_ylabel('Patch Row Index', fontsize=11)
ax1.set_title(f'Image Patch Masking (t=0)\n{num_patches_h}×{num_patches_w}={num_patches_img} patches\nValid: {num_valid_patches}, Masked: {img_patch_mask.sum()}/{num_valid_patches} ({img_patch_mask.sum()/num_valid_patches:.0%})',
              fontsize=12, fontweight='bold')
ax1.grid(False)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0.33, 1.0, 1.67])
cbar1.set_ticklabels(['Invalid\n(Ocean)', 'Visible\n(Land)', 'Masked\n(Land)'])

# Subplot 2: Zoomed view of image patches (top-left corner)
ax2 = plt.subplot(2, 3, 2)
zoom_h, zoom_w = 10, 10  # Show 10×10 patch region
zoom_state = patch_state[:zoom_h, :zoom_w]

# Create pixel-level visualization with 3 states
pixel_state = np.zeros((zoom_h * patch_size_img, zoom_w * patch_size_img))
for i in range(zoom_h):
    for j in range(zoom_w):
        pixel_state[i*patch_size_img:(i+1)*patch_size_img,
                   j*patch_size_img:(j+1)*patch_size_img] = zoom_state[i, j]

cmap_pixel = ListedColormap(['lightgray', 'lightblue', 'coral'])
im2 = ax2.imshow(pixel_state, cmap=cmap_pixel, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
ax2.set_xlabel('Pixel X', fontsize=11)
ax2.set_ylabel('Pixel Y', fontsize=11)
ax2.set_title(f'Zoomed Image Patches (top-left {zoom_h}×{zoom_w})\nEach patch = {patch_size_img}×{patch_size_img} pixels\nGray=Ocean, Blue=Visible, Red=Masked',
              fontsize=12, fontweight='bold')
ax2.grid(False)

# Draw grid lines to show patch boundaries
for i in range(zoom_h + 1):
    ax2.axhline(i * patch_size_img - 0.5, color='black', linewidth=1.5)
for j in range(zoom_w + 1):
    ax2.axvline(j * patch_size_img - 0.5, color='black', linewidth=1.5)

# Subplot 3: Temporal dimension (show masks over time)
ax3 = plt.subplot(2, 3, 3)
time_steps = 30  # Show 30 timesteps
# Only show valid patches in temporal view
time_mask_2d = np.zeros((time_steps, num_valid_patches), dtype=bool)
for t in range(time_steps):
    # Generate different random mask for each timestep (only from valid patches)
    num_to_mask_t = int(num_valid_patches * mask_ratio)
    masked_t = np.random.choice(num_valid_patches, size=num_to_mask_t, replace=False)
    time_mask_2d[t, masked_t] = True

cmap_temporal = ListedColormap(['lightblue', 'coral'])
im3 = ax3.imshow(time_mask_2d, cmap=cmap_temporal, aspect='auto', interpolation='nearest')
ax3.set_xlabel('Valid Patch Index', fontsize=11)
ax3.set_ylabel('Time Step', fontsize=11)
ax3.set_title(f'Image Patch Masking Over Time\n[T={time_steps}, valid_patches={num_valid_patches}]\nOnly showing land patches',
              fontsize=12, fontweight='bold')
ax3.grid(False)

# ===== Row 2: Vector Modality =====
# Subplot 4: Vector patches on geographic map
ax4 = plt.subplot(2, 3, 4)

# Color each catchment by its patch ID and mask status
patch_ids = np.arange(num_catchments) // patch_size_vec  # [604]
catchment_colors = np.zeros(num_catchments)
for i in range(num_catchments):
    patch_id = patch_ids[i]
    if vec_patch_mask[patch_id]:
        catchment_colors[i] = 1  # Masked
    else:
        catchment_colors[i] = 0  # Visible

scatter = ax4.scatter(lons, lats, c=catchment_colors, cmap=cmap_img, s=50,
                     alpha=0.7, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Longitude', fontsize=11)
ax4.set_ylabel('Latitude', fontsize=11)
ax4.set_title(f'Vector Patch Masking (Geographic View, t=0)\n{num_catchments} catchments → {num_patches_vec} patches\nMasked: {vec_patch_mask.sum()} patches ({mask_ratio:.0%})',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar4 = plt.colorbar(scatter, ax=ax4, ticks=[0.25, 0.75])
cbar4.set_ticklabels(['Visible', 'Masked'])

# Subplot 5: Vector patch layout (1D representation)
ax5 = plt.subplot(2, 3, 5)
# Create 2D representation of patches for better visualization
# Arrange patches in a grid (e.g., 19×4 for 76 patches)
grid_rows = 19
grid_cols = 4
vec_patch_grid = np.zeros((grid_rows, grid_cols), dtype=bool)
for i in range(num_patches_vec):
    row = i // grid_cols
    col = i % grid_cols
    if row < grid_rows:
        vec_patch_grid[row, col] = vec_patch_mask[i]

im5 = ax5.imshow(vec_patch_grid, cmap=cmap_img, aspect='auto', interpolation='nearest')
ax5.set_xlabel('Column', fontsize=11)
ax5.set_ylabel('Row', fontsize=11)
ax5.set_title(f'Vector Patch Layout (t=0)\n{grid_rows}×{grid_cols} grid, {num_patches_vec} patches\nEach patch = {patch_size_vec} catchments',
              fontsize=12, fontweight='bold')
ax5.grid(False)

# Add patch index annotations
for i in range(min(num_patches_vec, grid_rows * grid_cols)):
    row = i // grid_cols
    col = i % grid_cols
    if row < grid_rows:
        ax5.text(col, row, str(i), ha='center', va='center',
                fontsize=7, color='white' if vec_patch_grid[row, col] else 'black',
                fontweight='bold')

# Subplot 6: Vector temporal masking
ax6 = plt.subplot(2, 3, 6)
time_steps_vec = 30
vec_time_mask = np.zeros((time_steps_vec, num_patches_vec), dtype=bool)
for t in range(time_steps_vec):
    num_to_mask_t = int(num_patches_vec * mask_ratio)
    masked_t = np.random.choice(num_patches_vec, size=num_to_mask_t, replace=False)
    vec_time_mask[t, masked_t] = True

im6 = ax6.imshow(vec_time_mask, cmap=cmap_img, aspect='auto', interpolation='nearest')
ax6.set_xlabel('Patch Index', fontsize=11)
ax6.set_ylabel('Time Step', fontsize=11)
ax6.set_title(f'Vector Patch Masking Over Time\n[T={time_steps_vec}, num_patches={num_patches_vec}]',
              fontsize=12, fontweight='bold')
ax6.grid(False)

# Overall title
fig.suptitle('Patch-Level Masking Visualization: Image vs Vector Modalities',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_file = 'patch_masking_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ Visualization saved to: {output_file}")
print(f"{'='*70}")

# Show statistics
print(f"\nStatistics Summary:")
print(f"  Image Modality:")
print(f"    - Spatial coverage: {img_height}×{img_width} pixels")
print(f"    - Patchified into: {num_patches_h}×{num_patches_w} = {num_patches_img} patches")
print(f"    - Valid patches (land): {num_valid_patches}/{num_patches_img} ({num_valid_patches/num_patches_img:.1%})")
print(f"    - Invalid patches (ocean): {num_patches_img - num_valid_patches}/{num_patches_img} ({(num_patches_img - num_valid_patches)/num_patches_img:.1%})")
print(f"    - Masked patches: {img_patch_mask.sum()}/{num_valid_patches} ({img_patch_mask.sum()/num_valid_patches:.0%})")
print(f"\n  Vector Modality:")
print(f"    - Spatial coverage: {num_catchments} catchments across UK")
print(f"    - Patchified into: {num_patches_vec} patches ({patch_size_vec} catchments/patch)")
print(f"    - All patches are valid (no ocean)")
print(f"    - Masked patches: {vec_patch_mask.sum()}/{num_patches_vec} ({mask_ratio:.0%})")
print(f"\n  Masking Strategy:")
print(f"    - Image: Only mask from {num_valid_patches} VALID (land) patches")
print(f"    - Vector: Mask from all {num_patches_vec} patches (all valid)")
print(f"    - Both use {mask_ratio:.0%} mask ratio on VALID patches")
print(f"    - Independent random masks per timestep")
