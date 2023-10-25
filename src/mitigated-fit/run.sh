#!/bin/bash
#SBATCH --job-name=mitsgd
#SBATCH --output=mitigated_output_gluon_sim.log


python training.py gluon