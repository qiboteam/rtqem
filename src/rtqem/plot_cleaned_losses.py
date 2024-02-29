import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots
plt.style.use('science')

#updates = ["0.0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.2", "0.5"] 
updates = ["0.0", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "100"] 
changes = [99, 64, 8, 0]
#updates = ["0.0", "0.01", "0.1", "0.5", "1"]
indices = [0, 1, 3, 5]
spaces = ["    ", "", "", "  "]

losses = np.load("cleaned_losses.npy")
colors = sns.color_palette("magma", n_colors=len(indices)).as_hex()

m = min(losses[2])
mi = np.argmin(losses[2])

plt.figure(figsize=(5,5*6/8))
for i, idx in enumerate(indices):
    this_loss = losses[idx]
    min_value = min(this_loss)
    min_idx = np.argmin(this_loss)
    plt.plot(losses[idx], alpha=0.8, color=colors[i], lw=1.5, label=fr"$\varepsilon=${updates[idx]},{spaces[i]} $u$={changes[i]}")
for i, idx in enumerate(indices):
    this_loss = losses[idx]
    min_value = min(this_loss)
    min_idx = np.argmin(this_loss)
    plt.scatter(min_idx, min_value, color=colors[i], s=70, alpha=1)
plt.yscale("log")
plt.legend(loc=1, ncol=2, fontsize=10)
plt.xlabel("Epochs")
plt.ylabel("Cleaned loss")
plt.savefig("cleaned_losses.pdf", bbox_inches="tight")

mins = []
for i in range(len(updates)):
    mins.append(min(losses[i]))

x_line = np.arange(len(updates))