#! /usr/bin/bash

#SBATCH --job-name=testlog
#SBATCH --output=%x.o%j.out
#SBATCH --error=%x.o%j.err
#SBATCH --nodelist=pascal[23-24]
#SBATCH --partition=pvis
#SBATCH --wait-all-nodes=1


#launched on Node 1
srun --nodes=1 echo 'hello from node 1' > test.txt &

#Launched on Node2
srun --nodes=1 echo 'hello from node 2' >> test.txt &
