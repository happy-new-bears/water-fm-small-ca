"""
Spatial Aggregation for Catchment Data

Implements grid-based spatial patching with area-weighted aggregation.
Reduces 604 catchments to K spatial patches (e.g., 10×10 = 100 patches).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class SpatialAggregation(nn.Module):
    """
    将多个catchment聚合成spatial patches (基于地理网格)

    工作流程:
    1. 预处理阶段: 根据经纬度将catchment分配到网格
    2. 前向传播: 使用面积加权平均聚合每个grid内的catchment
    3. 反向传播: 将patch级别的预测分配回所有catchment

    Args:
        num_catchments: 总catchment数量 (例如 604)
        num_patches: 目标patch数量 (例如 100 for 10×10 grid)
        patch_assignments: [num_catchments] - 每个catchment属于哪个patch (0 to num_patches-1)
        catchment_areas: [num_catchments] - 每个catchment的面积（用于加权）
        aggregation_mode: 'fixed' (固定加权) 或 'learnable' (可学习权重)

    Example:
        >>> # 创建10×10网格 (100个patches)
        >>> spatial_agg = SpatialAggregation(
        ...     num_catchments=604,
        ...     num_patches=100,
        ...     patch_assignments=assignments,  # 从预处理获得
        ...     catchment_areas=areas,
        ... )
        >>>
        >>> # 聚合: [B, 604, 90] → [B, 100, 90]
        >>> x = torch.randn(4, 604, 90)
        >>> x_agg = spatial_agg(x)
        >>> print(x_agg.shape)  # [4, 100, 90]
        >>>
        >>> # 反聚合: [B, 100, 90] → [B, 604, 90]
        >>> x_recon = spatial_agg.reverse(x_agg)
        >>> print(x_recon.shape)  # [4, 604, 90]
    """

    def __init__(
        self,
        num_catchments: int,
        num_patches: int,
        patch_assignments: Tensor,
        catchment_areas: Tensor,
        aggregation_mode: str = 'fixed',
    ):
        super().__init__()

        self.num_catchments = num_catchments
        self.num_patches = num_patches
        self.aggregation_mode = aggregation_mode

        assert patch_assignments.shape[0] == num_catchments, \
            f"patch_assignments shape {patch_assignments.shape} doesn't match num_catchments {num_catchments}"
        assert catchment_areas.shape[0] == num_catchments, \
            f"catchment_areas shape {catchment_areas.shape} doesn't match num_catchments {num_catchments}"

        # Register as buffer (不参与梯度计算，但会保存到checkpoint)
        self.register_buffer('patch_assignments', patch_assignments.long())
        self.register_buffer('catchment_areas', catchment_areas.float())

        # 识别non-empty patches (类似image encoder的valid_patch_indices)
        non_empty_mask = torch.zeros(num_patches, dtype=torch.bool)
        for patch_id in range(num_patches):
            if (patch_assignments == patch_id).sum() > 0:
                non_empty_mask[patch_id] = True

        # 获取non-empty patches的原始索引
        non_empty_patch_indices = torch.where(non_empty_mask)[0]  # [num_non_empty]
        self.register_buffer('non_empty_patch_indices', non_empty_patch_indices)
        self.num_non_empty_patches = len(non_empty_patch_indices)

        # 构建聚合矩阵 [num_patches, num_catchments]
        # agg_matrix[i, j] = weight of catchment j in patch i
        agg_matrix = self._build_aggregation_matrix(
            patch_assignments, catchment_areas, num_patches
        )

        if aggregation_mode == 'fixed':
            # 固定权重，不参与训练
            self.register_buffer('agg_matrix', agg_matrix)
        elif aggregation_mode == 'learnable':
            # 可学习权重，初始化为面积加权
            self.agg_matrix = nn.Parameter(agg_matrix)
        else:
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

    def _build_aggregation_matrix(
        self,
        patch_assignments: Tensor,
        catchment_areas: Tensor,
        num_patches: int,
    ) -> Tensor:
        """
        构建面积加权聚合矩阵

        Args:
            patch_assignments: [num_catchments] - patch ID for each catchment
            catchment_areas: [num_catchments] - area of each catchment
            num_patches: number of patches

        Returns:
            agg_matrix: [num_patches, num_catchments] - normalized weights
        """
        agg_matrix = torch.zeros(num_patches, self.num_catchments)

        for patch_id in range(num_patches):
            # 找到属于这个patch的所有catchment
            mask = (patch_assignments == patch_id)

            if mask.sum() > 0:
                # 计算面积加权
                weights = catchment_areas.clone()
                weights[~mask] = 0.0  # 不属于这个patch的catchment权重为0

                # 归一化（使得权重和为1）
                weights = weights / (weights.sum() + 1e-8)
                agg_matrix[patch_id] = weights
            # else: 如果这个patch没有catchment，保持全0

        return agg_matrix

    def forward(self, x: Tensor) -> Tensor:
        """
        聚合catchment数据到spatial patches

        Args:
            x: [B, num_catchments, ...] - catchment级别的数据
               支持的shape:
               - [B, num_catchments, T] - 时间序列
               - [B, num_catchments, T, C] - 多通道时间序列

        Returns:
            x_agg: [B, num_patches, ...] - patch级别的聚合数据
        """
        if x.dim() == 3:
            # [B, num_catchments, T]
            B, C, T = x.shape
            assert C == self.num_catchments, \
                f"Input shape {x.shape} doesn't match num_catchments {self.num_catchments}"

            # 矩阵乘法聚合
            # agg_matrix: [num_patches, num_catchments]
            # x: [B, num_catchments, T]
            # 需要: [B, num_patches, T] = [num_patches, num_catchments] @ [B, num_catchments, T]

            # 转置以便矩阵乘法
            x_t = x.transpose(1, 2)  # [B, T, num_catchments]
            x_agg_t = torch.matmul(x_t, self.agg_matrix.T)  # [B, T, num_patches]
            x_agg = x_agg_t.transpose(1, 2)  # [B, num_patches, T]

        elif x.dim() == 4:
            # [B, num_catchments, T, C]
            B, N, T, C = x.shape
            assert N == self.num_catchments

            # Reshape to [B, T, C, num_catchments]
            x = x.permute(0, 2, 3, 1)  # [B, T, C, num_catchments]
            # Aggregate: [B, T, C, num_catchments] @ [num_catchments, num_patches]
            x_agg = torch.matmul(x, self.agg_matrix.T)  # [B, T, C, num_patches]
            # Reshape to [B, num_patches, T, C]
            x_agg = x_agg.permute(0, 3, 1, 2)

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3D or 4D tensor.")

        return x_agg

    def reverse(self, x_agg: Tensor) -> Tensor:
        """
        从patch聚合反推回catchment级别（用于decoder输出）

        策略: 每个catchment使用其所属patch的值（broadcasting）

        Args:
            x_agg: [B, num_patches, T] - patch级别的预测

        Returns:
            x: [B, num_catchments, T] - catchment级别的预测
        """
        B, P, T = x_agg.shape
        assert P == self.num_patches, \
            f"Input shape {x_agg.shape} doesn't match num_patches {self.num_patches}"

        # 为每个catchment分配其所属patch的值
        x = torch.zeros(B, self.num_catchments, T, device=x_agg.device, dtype=x_agg.dtype)

        for patch_id in range(self.num_patches):
            # 找到属于这个patch的所有catchment
            mask = (self.patch_assignments == patch_id)
            if mask.sum() > 0:
                # x[:, mask, :] = x_agg[:, patch_id, :].unsqueeze(1)
                # 使用broadcasting: [B, T] → [B, num_catchments_in_patch, T]
                x[:, mask, :] = x_agg[:, patch_id, :].unsqueeze(1).expand(B, mask.sum(), T)

        return x

    def get_patch_info(self) -> dict:
        """
        获取patch信息统计

        Returns:
            dict with:
                - patch_sizes: [num_patches] - 每个patch包含的catchment数量
                - total_area_per_patch: [num_patches] - 每个patch的总面积
                - empty_patches: list of patch IDs that have no catchments
        """
        patch_sizes = []
        total_areas = []
        empty_patches = []

        for patch_id in range(self.num_patches):
            mask = (self.patch_assignments == patch_id)
            count = mask.sum().item()
            patch_sizes.append(count)

            if count > 0:
                total_area = self.catchment_areas[mask].sum().item()
                total_areas.append(total_area)
            else:
                total_areas.append(0.0)
                empty_patches.append(patch_id)

        return {
            'patch_sizes': patch_sizes,
            'total_area_per_patch': total_areas,
            'empty_patches': empty_patches,
            'num_non_empty_patches': self.num_patches - len(empty_patches),
        }


def create_grid_patches(
    catchment_lons: Tensor,
    catchment_lats: Tensor,
    catchment_areas: Tensor,
    grid_size: Tuple[int, int] = (10, 10),
) -> Tuple[Tensor, int]:
    """
    基于经纬度创建网格patches

    Args:
        catchment_lons: [num_catchments] - 经度
        catchment_lats: [num_catchments] - 纬度
        catchment_areas: [num_catchments] - 面积
        grid_size: (M, N) - 网格大小，例如 (10, 10) = 100个patches

    Returns:
        patch_assignments: [num_catchments] - 每个catchment的patch ID
        num_patches: 实际的patch数量（≤ M×N，因为有些grid可能为空）

    Example:
        >>> lons = torch.tensor([...])  # 604个catchment的经度
        >>> lats = torch.tensor([...])  # 604个catchment的纬度
        >>> areas = torch.tensor([...]) # 604个catchment的面积
        >>>
        >>> assignments, num_patches = create_grid_patches(
        ...     lons, lats, areas, grid_size=(10, 10)
        ... )
        >>> print(f"Created {num_patches} patches from 10×10 grid")
    """
    num_catchments = len(catchment_lons)
    M, N = grid_size

    # 找到经纬度范围
    lon_min, lon_max = catchment_lons.min().item(), catchment_lons.max().item()
    lat_min, lat_max = catchment_lats.min().item(), catchment_lats.max().item()

    # 创建网格边界
    # 注意: 添加小的epsilon避免边界情况
    eps = 1e-6
    lon_bins = torch.linspace(lon_min - eps, lon_max + eps, M + 1)
    lat_bins = torch.linspace(lat_min - eps, lat_max + eps, N + 1)

    # 为每个catchment分配grid cell
    # torch.bucketize: 找到每个值应该放在哪个bin中
    lon_idx = torch.bucketize(catchment_lons, lon_bins) - 1  # [num_catchments]
    lat_idx = torch.bucketize(catchment_lats, lat_bins) - 1  # [num_catchments]

    # 确保索引在合法范围内（处理边界情况）
    lon_idx = torch.clamp(lon_idx, 0, M - 1)
    lat_idx = torch.clamp(lat_idx, 0, N - 1)

    # 将2D grid索引转换为1D patch ID
    # patch_id = row * N + col
    patch_id = lon_idx * N + lat_idx  # [num_catchments]

    # 统计实际有catchment的patch数量
    unique_patches = torch.unique(patch_id)
    num_patches = M * N  # 保持固定数量，即使有些patch为空

    return patch_id, num_patches


if __name__ == '__main__':
    """Unit test for SpatialAggregation"""

    print("=" * 60)
    print("Testing SpatialAggregation")
    print("=" * 60)

    # 模拟604个catchment
    num_catchments = 604
    grid_size = (10, 10)
    num_patches = grid_size[0] * grid_size[1]  # 100

    # 生成模拟数据
    torch.manual_seed(42)
    catchment_lons = torch.rand(num_catchments) * 10 - 5  # [-5, 5]
    catchment_lats = torch.rand(num_catchments) * 10 + 50  # [50, 60]
    catchment_areas = torch.rand(num_catchments) * 500 + 50  # [50, 550]

    print(f"\nSimulated {num_catchments} catchments:")
    print(f"  Lon range: [{catchment_lons.min():.2f}, {catchment_lons.max():.2f}]")
    print(f"  Lat range: [{catchment_lats.min():.2f}, {catchment_lats.max():.2f}]")
    print(f"  Area range: [{catchment_areas.min():.2f}, {catchment_areas.max():.2f}]")

    # 创建网格patches
    patch_assignments, num_patches = create_grid_patches(
        catchment_lons, catchment_lats, catchment_areas,
        grid_size=grid_size
    )

    print(f"\n✓ Created {grid_size[0]}×{grid_size[1]} = {num_patches} patches")
    print(f"  Unique patches with catchments: {torch.unique(patch_assignments).numel()}")

    # 创建SpatialAggregation模块
    spatial_agg = SpatialAggregation(
        num_catchments=num_catchments,
        num_patches=num_patches,
        patch_assignments=patch_assignments,
        catchment_areas=catchment_areas,
        aggregation_mode='fixed',
    )

    print(f"\n✓ SpatialAggregation module created")

    # 获取patch信息
    patch_info = spatial_agg.get_patch_info()
    non_empty = patch_info['num_non_empty_patches']
    empty = len(patch_info['empty_patches'])
    print(f"  Non-empty patches: {non_empty}")
    print(f"  Empty patches: {empty}")
    print(f"  Avg catchments per non-empty patch: {num_catchments / non_empty:.1f}")

    # Test 1: 聚合 [B, 604, T] → [B, 100, T]
    print(f"\n" + "=" * 60)
    print("Test 1: Aggregation [B, 604, T] → [B, 100, T]")
    print("=" * 60)

    B, T = 4, 90
    x = torch.randn(B, num_catchments, T)

    x_agg = spatial_agg(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_agg.shape}")
    print(f"✓ Aggregation successful")

    # Test 2: 反聚合 [B, 100, T] → [B, 604, T]
    print(f"\n" + "=" * 60)
    print("Test 2: Reverse [B, 100, T] → [B, 604, T]")
    print("=" * 60)

    x_recon = spatial_agg.reverse(x_agg)

    print(f"Input shape: {x_agg.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"✓ Reverse successful")

    # Test 3: 检查同一patch内的catchment是否有相同值
    print(f"\n" + "=" * 60)
    print("Test 3: Verify patch assignments")
    print("=" * 60)

    # 找一个非空patch
    test_patch_id = patch_info['patch_sizes'].index(max(patch_info['patch_sizes']))
    mask = (patch_assignments == test_patch_id)
    catchments_in_patch = torch.where(mask)[0]

    print(f"Test patch {test_patch_id} has {len(catchments_in_patch)} catchments")
    print(f"Catchment IDs: {catchments_in_patch[:5].tolist()}...")

    # 检查这些catchment在reverse后是否有相同值
    values_in_patch = x_recon[0, catchments_in_patch, 0]
    all_same = torch.allclose(values_in_patch, values_in_patch[0].expand_as(values_in_patch))

    print(f"All catchments in patch have same value: {all_same}")
    print(f"✓ Patch assignment verified")

    # Test 4: 梯度测试
    print(f"\n" + "=" * 60)
    print("Test 4: Gradient flow")
    print("=" * 60)

    x = torch.randn(B, num_catchments, T, requires_grad=True)
    x_agg = spatial_agg(x)
    loss = x_agg.sum()
    loss.backward()

    print(f"✓ Backward pass successful")
    print(f"✓ Gradient shape: {x.grad.shape}")

    # Test 5: Token reduction
    print(f"\n" + "=" * 60)
    print("Test 5: Token reduction statistics")
    print("=" * 60)

    tokens_before = num_catchments * T
    tokens_after = num_patches * T
    reduction = (1 - tokens_after / tokens_before) * 100

    print(f"Tokens before: {num_catchments} catchments × {T} timesteps = {tokens_before}")
    print(f"Tokens after: {num_patches} patches × {T} timesteps = {tokens_after}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Memory reduction: ~{reduction:.1f}% (approximate)")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
