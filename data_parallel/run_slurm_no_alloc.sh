#! /usr/bin/bash

#SBATCH --job-name=pytorch_ddp
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=pvis

srun /g/g11/eisenbnt/venvs/base/bin/python3 \
	        -u /g/g11/eisenbnt/projects/data_parallel/main.py
