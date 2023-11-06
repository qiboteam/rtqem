import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

updates = ["000", "1e-4", "1e-3", "5e-3", "1e-2", "2.5e-2", "5e-2", "7.5e-2", "1e-1", "1.5e-1", "2e-1", "3e-1", "4e-1"]
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

losses = np.load("cleaned_losses.npy")
colors = sns.color_palette("magma", n_colors=len(updates)).as_hex()

m = min(losses[2])
mi = np.argmin(losses[2])

plt.figure(figsize=(5,5*6/8))
for i, idx in enumerate(indices):
    this_loss = losses[idx][0:50]
    min_value = min(this_loss)
    min_idx = np.argmin(this_loss)
    plt.plot(losses[idx][0:50], alpha=0.7, color=colors[idx], lw=2, label=f"Threshold: {updates[idx]}")
for i, idx in enumerate(indices):
    this_loss = losses[idx][0:50]
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