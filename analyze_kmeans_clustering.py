"""
KMeans Clustering Analysis for 604 Catchments
使用KMeans聚类方法对catchments进行空间分组
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# File paths
attr_file = '/Users/transformer/Desktop/water_data/Catchment_attributes/Catchment_attributes_nrfa.csv'
parquet_file = '/Users/transformer/Desktop/water_data/new_version/riverflow_evaporation_604catchments_1970_2015.parquet'

print("=" * 70)
print("KMeans Clustering Analysis for 604 Catchments")
print("=" * 70)

# Load catchment IDs from parquet
print(f"\nLoading catchment IDs from: {parquet_file}")
df_parquet = pd.read_parquet(parquet_file)
catchment_ids_604 = df_parquet['ID'].unique()
catchment_ids_604 = np.sort(catchment_ids_604)
print(f"✓ Found {len(catchment_ids_604)} catchments")

# Load attributes
print(f"\nLoading catchment attributes from: {attr_file}")
df_attr = pd.read_csv(attr_file)
print(f"✓ Total catchments in CSV: {len(df_attr)}")

# Filter to only 604 catchments
df_604 = df_attr[df_attr['id'].isin(catchment_ids_604)].copy()
print(f"✓ Filtered to {len(df_604)} catchments")

# Extract coordinates
lats = df_604['latitude'].values
lons = df_604['longitude'].values
ids = df_604['id'].values

# Prepare data for KMeans (lat, lon)
X = np.column_stack([lats, lons])

# Try different numbers of clusters
n_clusters_list = [50, 100, 150]

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

results = {}

for idx, n_clusters in enumerate(n_clusters_list):
    print(f"\n{'='*70}")
    print(f"KMeans Clustering with {n_clusters} clusters")
    print(f"{'='*70}")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    # Statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n✓ Clustering completed")
    print(f"  Number of clusters: {len(unique_clusters)}")
    print(f"  Catchments per cluster: min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}, std={counts.std():.2f}")

    # Distribution statistics
    print(f"\n  Distribution of catchments per cluster:")
    print(f"    0-3 catchments: {np.sum(counts <= 3)} clusters")
    print(f"    4-6 catchments: {np.sum((counts >= 4) & (counts <= 6))} clusters")
    print(f"    7-10 catchments: {np.sum((counts >= 7) & (counts <= 10))} clusters")
    print(f"    >10 catchments: {np.sum(counts > 10)} clusters")

    results[n_clusters] = {
        'labels': cluster_labels,
        'centers': cluster_centers,
        'counts': counts
    }

    # Visualization 1: Geographic map with cluster colors
    ax1 = axes[idx]
    scatter = ax1.scatter(lons, lats, c=cluster_labels, cmap='tab20',
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.scatter(cluster_centers[:, 1], cluster_centers[:, 0],
               c='red', marker='X', s=200, edgecolors='black', linewidth=2,
               label='Cluster Centers')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'KMeans Clustering (n={n_clusters})\n604 Catchments Spatial Distribution',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Visualization 2: Histogram of catchments per cluster
    ax2 = axes[idx + 3]
    ax2.hist(counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(counts.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {counts.mean():.2f}')
    ax2.axvline(counts.min(), color='green', linestyle='--', linewidth=2,
                label=f'Min: {counts.min()}')
    ax2.axvline(counts.max(), color='orange', linestyle='--', linewidth=2,
                label=f'Max: {counts.max()}')
    ax2.set_xlabel('Number of Catchments per Cluster', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title(f'Distribution of Catchments per Cluster (n={n_clusters})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'catchment_kmeans_clustering_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Save the best clustering result (100 clusters) to file
print(f"\n{'='*70}")
print("Saving clustering results (n=100 clusters)")
print(f"{'='*70}")

best_n_clusters = 100
cluster_labels = results[best_n_clusters]['labels']

# Create mapping: catchment_id -> cluster_id
cluster_mapping = {}
for catchment_id, cluster_id in zip(ids, cluster_labels):
    cluster_mapping[int(catchment_id)] = int(cluster_id)

# Save as PyTorch tensor for easy use in training
# Format: [catchment_id, cluster_id, lat, lon]
cluster_data = np.column_stack([ids, cluster_labels, lats, lons])
cluster_tensor = torch.from_numpy(cluster_data).float()

output_tensor_file = 'data/catchment_kmeans_clusters_100.pt'
torch.save({
    'cluster_assignments': cluster_tensor,  # [604, 4]: [id, cluster, lat, lon]
    'cluster_mapping': cluster_mapping,      # dict: {catchment_id: cluster_id}
    'n_clusters': best_n_clusters,
    'cluster_centers': results[best_n_clusters]['centers'],
    'catchments_per_cluster': results[best_n_clusters]['counts']
}, output_tensor_file)

print(f"✓ Cluster data saved to: {output_tensor_file}")
print(f"\n  Format:")
print(f"    - cluster_assignments: [604, 4] tensor with [id, cluster, lat, lon]")
print(f"    - cluster_mapping: dict mapping catchment_id -> cluster_id")
print(f"    - n_clusters: {best_n_clusters}")
print(f"    - cluster_centers: [{best_n_clusters}, 2] array with [lat, lon]")
print(f"    - catchments_per_cluster: [{best_n_clusters}] array with counts")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
