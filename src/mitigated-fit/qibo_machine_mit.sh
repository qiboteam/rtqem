#!/bin/bash

#SBATCH --job-name=mit_bp
#SBATCH --partition=sim
#SBATCH --output=latest_experiment1.log

#SBATCH --cpus-per-task=150
#SBATCH --mem=500000

python stat-on-result.py cosnd --run_name benchmark
#python training.py cosnd