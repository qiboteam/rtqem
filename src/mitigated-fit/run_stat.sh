#!/bin/bash
#SBATCH --job-name=stat_sgd
#SBATCH --output=stat.log


# mkdir ./gluon/sgd
# mkdir ./gluon/sgd/cache

# cp ./gluon/gluon.conf ./gluon/sgd/gluon.conf

# cp ./gluon/cache/best_params_Adam_noiseless.npy ./gluon/sgd/cache/best_params_Adam_realtime_mitigation_step_yes_final_yes.npy
# cp ./gluon/cache/grad_history_noiseless.npy ./gluon/sgd/cache/grad_history_realtime_mitigation_step_yes_final_yes.npy
# cp ./gluon/cache/loss_history_noiseless.npy ./gluon/sgd/cache/loss_history_realtime_mitigation_step_yes_final_yes.npy

# cp -r ./gluon/cache/params_history_noiseless ./gluon/sgd/cache/params_history_realtime_mitigation_step_yes_final_yes

python stat-alej.py gluon --run_name sgd 
