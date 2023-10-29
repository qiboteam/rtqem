#!/bin/bash
#SBATCH --job-name=stat_sgd
#SBATCH --output=stat.log


# mkdir ./gluon/sgd_long
# mkdir ./gluon/sgd_long/cache

# cp ./gluon/gluon.conf ./gluon/sgd_long/gluon.conf

# cp ./gluon/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_long/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy
# cp ./gluon/cache/grad_history_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_long/cache/grad_history_full_mitigation_step_yes_final_yes.npy
# cp ./gluon/cache/loss_history_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_long/cache/loss_history_full_mitigation_step_yes_final_yes.npy

# cp -r ./gluon/cache/params_history_full_mitigation_step_yes_final_yes ./gluon/sgd_long/cache/params_history_full_mitigation_step_yes_final_yes

python stat-on-result.py gluon --run_name sgd_long 
