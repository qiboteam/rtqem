# some useful python package
import numpy as np
from vqregressor import vqregressor
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates

ndata = 10
# random data
data = np.random.uniform(-1, 1, ndata)
# labeling them
labels = np.sin(2*data)
# noise model
noise = NoiseModel()
noise.add(DepolarizingError(lam=0.25), gates.RZ)

zne = ('ZNE', {'noise_levels':np.arange(5), 'insertion_gate':'RX'})
cdr = ('CDR', {'n_training_samples':10})
vncdr = ('vnCDR', {'n_training_samples':10, 'noise_levels':np.arange(3), 'insertion_gate':'RX'})

VQR = vqregressor(
    layers=1,
    data=data,
    labels=labels,
    noise_model=noise,
    mitigation=cdr[0],
    mit_kwargs=cdr[1]
)
# set the training hyper-parameters
epochs = 50
learning_rate = 0.1

# perform the training
history = VQR.gradient_descent(learning_rate=learning_rate, epochs=epochs)
VQR.show_predictions('predictions_psr', save=True)

np.save("results/best_params_psr", VQR.params)
