#! /usr/bin/bash

#SBATCH --job-name=pytorch_ddp
#SBATCH --output=%x.o%j

srun --jobid=718612 -w pascal27 /g/g11/eisenbnt/venvs/base/bin/python3 \
	-u /g/g11/eisenbnt/projects/data_parallel/main.py
