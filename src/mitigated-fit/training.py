# some useful python package
import numpy as np
import scipy.stats
from vqregressor import vqregressor
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates

ndata = 500
# random data
data = np.random.uniform(-1, 1, ndata)
# labeling them
labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.3)

# noise model
noise = NoiseModel()
noise.add(DepolarizingError(lam=0.25), gates.RZ)

VQR = vqregressor(
    layers=6, 
    data=data, 
    labels=labels, 
    noise_model=None
    )

# set the training hyper-parameters
epochs = 15
learning_rate = 5e-2

# perform the training
history = VQR.gradient_descent(
    learning_rate=learning_rate, 
    epochs=epochs, 
    restart_from_epoch=None,
    batchsize=50,
    method='Adam'
    )

VQR.show_predictions('predictions_psr', save=True)

np.save("results/best_params_psr", VQR.params)