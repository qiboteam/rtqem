#!/bin/bash
#SBATCH --job-name=mitsgd
#SBATCH --output=long_gluon_run.log


python training.py gluon