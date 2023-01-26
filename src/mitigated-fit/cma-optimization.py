# some useful python package
import numpy as np
from vqregressor import vqregressor

ndata = 30
# random data
data = np.random.uniform(-1, 1, ndata)
# labeling them
#labels = np.sin(2*data)
labels = (2*data+2)**2*np.exp(-(2*data+2)**2/2) 

VQR = vqregressor(layers=3, data=data, labels=labels, nshots=1000, expectation_from_samples=False)

# perform the training
result, best_params = VQR.cma_optimization()
VQR.show_predictions('predictions_cma', save=True)

# save best params
np.save("results/best_params_cma", best_params)
