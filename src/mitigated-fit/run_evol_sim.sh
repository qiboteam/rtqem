#!/bin/bash
#SBATCH --job-name=evo_000
#SBATCH --output=sim_evolution_000.log

name=000

python training.py cosnd

mkdir ./cosnd/evolution
mkdir ./cosnd/evolution/evol_$name
mkdir ./cosnd/evolution/evol_$name/cache

cp ./cosnd/cosnd.conf ./cosnd/evolution/evol_$name/cosnd.conf

cp ./cosnd/cache/best_params_Adam_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/best_params_Adam_realtime_mitigation_step_yes_final_yes.npy
cp ./cosnd/cache/grad_history_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/grad_history_realtime_mitigation_step_yes_final_yes.npy
cp ./cosnd/cache/loss_history_realtime_mitigation_step_yes_final_yes.npy ./cosnd/evolution/evol_$name/cache/loss_history_realtime_mitigation_step_yes_final_yes.npy

cp -r ./cosnd/cache/params_history_realtime_mitigation_step_yes_final_yes ./cosnd/evolution/evol_$name/cache/params_history_realtime_mitigation_step_yes_final_yes

