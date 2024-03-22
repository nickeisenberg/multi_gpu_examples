#!/bin/bash

#SBATCH --job-name=pytorch_ddp
#SBATCH --output=%x.o%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=pvis

# Directly using the hostname for MASTER_ADDR
MASTER_ADDR=$(hostname)

# Setup environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=12345
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES='0,1'

# Run the PyTorch script
srun /g/g11/eisenbnt/venvs/base/bin/python3 \
    -u /g/g11/eisenbnt/projects/distributed_data_parallel/main.py 
