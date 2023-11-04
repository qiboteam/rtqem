import numpy as np
import matplotlib.pyplot as plt

radii = np.load("noise_radii.npy")
bounds = np.load("bounds.npy")

plt.figure(figsize=(8,4*6/8))
plt.subplot(1,2,1)
plt.plot(radii)
plt.title("Noise radius")
plt.xlabel("Epochs")
plt.ylabel("r")

plt.subplot(1,2,2)
plt.plot(bounds)
plt.title("Bound")
plt.xlabel("Epochs")
plt.ylabel("BP bound")
plt.savefig("noise_analysis.png", bbox_inches="tight")

