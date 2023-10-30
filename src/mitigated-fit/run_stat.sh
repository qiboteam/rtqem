#!/bin/bash
#SBATCH --job-name=stat_sgd
#SBATCH --output=stat_gorka.log


mkdir ./hdw_target/sgd_j_2_target
mkdir ./hdw_target/sgd_j_2_target/cache

cp ./hdw_target/hdw_target.conf ./hdw_target/sgd_j_2_target/hdw_target.conf

cp ./hdw_target/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy ./hdw_target/sgd_j_2_target/cache/best_params_Adam_full_mitigation_step_yes_final_yes.npy
cp ./hdw_target/cache/grad_history_full_mitigation_step_yes_final_yes.npy ./hdw_target/sgd_j_2_target/cache/grad_history_full_mitigation_step_yes_final_yes.npy
cp ./hdw_target/cache/loss_history_full_mitigation_step_yes_final_yes.npy ./hdw_target/sgd_j_2_target/cache/loss_history_full_mitigation_step_yes_final_yes.npy

cp -r ./hdw_target/cache/params_history_full_mitigation_step_yes_final_yes ./hdw_target/sgd_j_2_target/cache/params_history_full_mitigation_step_yes_final_yes

python stat-on-result.py hdw_target --run_name sgd_j_2_target 
