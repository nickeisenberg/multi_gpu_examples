#! /usr/bin/bash

srun --qos=standby \
	-p pvis \
	-N 1 \
	-n 1 \
	/g/g11/eisenbnt/projects/figure_out_llnl/main/main.sh &
