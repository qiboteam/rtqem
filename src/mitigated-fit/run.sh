#!/bin/bash
#SBATCH --job-name=rtqem
#SBATCH --output=rtqem.log

python training.py uquark
