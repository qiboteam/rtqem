#!/bin/bash
#SBATCH --job-name=stat_sgd
#SBATCH --output=sim_evolution_Inf.log


mkdir ./gluon/sgd_iqm5q
mkdir ./gluon/sgd_iqm5q/cache

cp ./gluon/gluon.conf ./gluon/sgd_iqm5q/gluon.conf

cp ./gluon/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_iqm5q/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy
cp ./gluon/cache/grad_history_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_iqm5q/cache/grad_history_full_mitigation_step_yes_final_yes.npy
cp ./gluon/cache/loss_history_full_mitigation_step_yes_final_yes.npy ./gluon/sgd_iqm5q/cache/loss_history_full_mitigation_step_yes_final_yes.npy

cp ./gluon/cache/best_params_Adam_noiseless.npy ./gluon/sgd_iqm5q/cache/best_params_Adam_noiseless.npy
cp ./gluon/cache/grad_history_noiseless.npy ./gluon/sgd_iqm5q/cache/grad_history_noiseless.npy
cp ./gluon/cache/loss_history_noiseless.npy ./gluon/sgd_iqm5q/cache/loss_history_noiseless.npy

cp -r ./gluon/cache/params_history_full_mitigation_step_yes_final_yes ./gluon/sgd_iqm5q/cache/params_history_full_mitigation_step_yes_final_yes
cp -r ./gluon/cache/params_history_noiseless ./gluon/sgd_iqm5q/cache/params_history_noiseless

python stat-on-result.py gluon --run_name sgd_iqm5q 
