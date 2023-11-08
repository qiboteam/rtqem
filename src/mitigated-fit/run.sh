#!/bin/bash
#SBATCH --job-name=myjob_name
#SBATCH --output=myjob.log

python training.py cosnd
