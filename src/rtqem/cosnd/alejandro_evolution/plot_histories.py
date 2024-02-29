import os

import numpy as np
import matplotlib.pyplot as plt

losses = []
updates = ["Inf", "035", "03", "025", "02", "015", "01", "005"]

plt.figure(figsize=(7,7*6/8))

for i, run in enumerate(updates):
    losses.append(np.load(f"runs/4q_update{run}/cache/loss_history_realtime_mitigation_step_yes_final_yes.npy"))
    plt.plot(losses[-1], color="royalblue", alpha=1-0.1*i, label=f"Treshold {run}")

plt.yscale("log")
plt.legend()
plt.savefig("losses_evolution.png")