#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=tii1q_b1

python stat-on-results.py --best_params_path results/best_params_psr.npy

