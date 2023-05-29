#!/bin/bash

#SBATCH --job-name=36_q
#SBATCH --partition=tii1q_b1
#SBATCH --output=latest_experiment1.log


kernprof -lv training.py gluon
