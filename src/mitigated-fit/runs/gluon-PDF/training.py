# some useful python package
import numpy as np
from vqregressor import vqregressor
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates
import random

# model definition
nqubits = 1
layers = 4

ndata = 50
parton = 'u'

data = np.loadtxt(f'data/{parton}.dat')

idx = random.sample(range(len(data)), ndata)

x = data.T[0][idx]
y = data.T[1][idx]

# noise model
noise = NoiseModel()
noise.add(DepolarizingError(lam=0.25), gates.RZ)

VQR = vqregressor(
    layers=layers, 
    data=x, 
    labels=y, 
    noise_model=None
    )

# set the training hyper-parameters
epochs = 500
learning_rate = 2.5e-2

# perform the training
history = VQR.gradient_descent(
    learning_rate=learning_rate, 
    epochs=epochs, 
    restart_from_epoch=None,
    batchsize=ndata,
    method='Adam'
    )

VQR.show_predictions('predictions_psr', save=True)

np.save("results/best_params_psr", VQR.params)
np.save("results/model_info_psr", np.array([nqubits, layers]))
