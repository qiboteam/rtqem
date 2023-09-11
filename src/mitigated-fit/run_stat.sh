#!/bin/bash
#SBATCH --job-name=stat_sgd

python stat-on-result.py gluon --run_name benchmark
