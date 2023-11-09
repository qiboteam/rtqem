#!/bin/bash
#SBATCH --job-name=stat_sgd
#SBATCH --output=stat_on_result.log


python stat_on_result.py uquark --run_name results_folder
