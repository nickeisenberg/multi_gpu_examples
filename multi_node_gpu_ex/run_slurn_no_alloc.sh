#! /usr/bin/bash

#SBATCH --job-name=testlog
#SBATCH --output=%x.o%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=pvis
               
srun /g/g11/eisenbnt/venvs/base/bin/python3 \
    -u /g/g11/eisenbnt/projects/multi_node_gpu_ex/main.py
