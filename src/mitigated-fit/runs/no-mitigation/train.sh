#!/bin/bash
#SBATCH --job-name=train_fit
#SBATCH --partition=tii1q_b1

QIBO_LOG_LEVEL=3 python training.py

