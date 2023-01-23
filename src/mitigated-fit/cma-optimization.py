# some useful python package
import numpy as np
from vqregressor import vqregressor

ndata = 10
# random data
data = np.random.uniform(-1, 1, ndata)
# labeling them
labels = np.sin(2*data)

VQR = vqregressor(layers=1, data=data, labels=labels)

# perform the training
result, best_params = VQR.cma_optimization()
VQR.show_predictions('predictions_cma', save=True)

# save best params
np.save("results/best_params_cma", best_params)
