import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

updates = ["Inf", "0.35", "0.3", "0.25", "0.2", "0.15", "0.1", "0.05", "0"]
indices = [0, 2, 4, 6, 8]

losses = np.load("cleaned_losses.npy")
colors = sns.color_palette("magma", n_colors=len(updates)).as_hex()

m = min(losses[2])
mi = np.argmin(losses[2])

plt.figure(figsize=(5,5*6/8))
for i, idx in enumerate(indices):
    this_loss = losses[idx]
    min_value = min(this_loss)
    min_idx = np.argmin(this_loss)
    plt.plot(losses[idx], alpha=0.7, color=colors[idx], lw=2, label=f"Treshold: {updates[idx]}")
for i, idx in enumerate(indices):
    this_loss = losses[idx]
    min_value = min(this_loss)
    min_idx = np.argmin(this_loss)
    plt.scatter(min_idx, min_value, color=colors[idx], s=70, alpha=1)
plt.yscale("log")
plt.legend(loc=1, ncol=2)
plt.savefig("cleaned_losses.png")

mins = []
for i in range(len(updates)):
    mins.append(min(losses[i]))

x_line = np.arange(len(updates))