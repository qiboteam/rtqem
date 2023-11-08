#!/bin/bash
#SBATCH --job-name=ev_100
#SBATCH --output=sim_evolution_100.log

name=100

python training.py cosnd

mkdir ./cosnd/evolution
mkdir ./cosnd/evolution/evol_$name
mkdir ./cosnd/evolution/evol_$name/cache

cp ./cosnd/cosnd.conf ./cosnd/evolution/evol_$name/cosnd.conf

cp ./cosnd/cache_$name/best_params_Adam_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/best_params_Adam_realtime_mitigation_step_yes_final_yes.npy
cp ./cosnd/cache_$name/grad_history_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/grad_history_realtime_mitigation_step_yes_final_yes.npy
cp ./cosnd/cache_$name/loss_history_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/loss_history_realtime_mitigation_step_yes_final_yes.npy

cp -r ./cosnd/cache_$name/params_history_realtime_mitigation_step_yes_final_yes ./cosnd/evolution/evol_$name/cache/params_history_realtime_mitigation_step_yes_final_yes

