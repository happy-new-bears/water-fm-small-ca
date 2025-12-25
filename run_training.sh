#!/bin/bash
# Multi-GPU Training Script for Jupyter Terminal on HPC
# Usage: bash run_training.sh

echo "=========================================="
echo "Multi-modal MAE Training with DeepSpeed"
echo "=========================================="

# Check available GPUs
echo -e "\nChecking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "\nFound ${NUM_GPUS} GPUs"

# Set number of GPUs to use (modify this if you want to use fewer GPUs)
NUM_GPUS_TO_USE=${1:-${NUM_GPUS}}  # Use argument or all available GPUs
echo "Using ${NUM_GPUS_TO_USE} GPUs for training"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Modify based on your needs
export NCCL_DEBUG=WARN  # Set to INFO for more detailed logs
export PYTHONUNBUFFERED=1  # Ensure real-time output

# Set a custom master port to avoid conflicts
MASTER_PORT=${2:-29500}
echo "Using master port: ${MASTER_PORT}"

# Navigate to project directory
cd /Users/transformer/Desktop/water_code/water_fm

echo -e "\n=========================================="
echo "Starting training..."
echo "=========================================="

# Launch training with DeepSpeed
deepspeed --num_gpus=${NUM_GPUS_TO_USE} \
          --master_port=${MASTER_PORT} \
          train_mae.py

# Alternative: Use torchrun
# torchrun --nproc_per_node=${NUM_GPUS_TO_USE} \
#          --master_port=${MASTER_PORT} \
#          train_mae.py

echo -e "\n=========================================="
echo "Training completed or interrupted"
echo "=========================================="
