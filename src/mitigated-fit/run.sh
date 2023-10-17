#!/bin/bash
#SBATCH --job-name=sgd

QIBO_LOG_LEVEL=4 python training.py gluon
