#!/bin/bash

# DeepSpeed安装脚本（在HPC上使用）

echo "正在加载必要的模块..."

# 加载PyTorch和CUDA环境
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# 验证环境
echo "检查CUDA环境..."
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc路径: $(which nvcc)"
nvcc --version

# 检查PyTorch CUDA是否可用
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "开始安装DeepSpeed..."
echo ""

# 方法1: 完整安装（带CUDA算子编译）
# pip install deepspeed

# 方法2: 快速安装（不编译CUDA算子，运行时JIT编译）
DS_BUILD_OPS=0 pip install deepspeed

echo ""
echo "安装完成！验证安装..."
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
