# some useful python package
import numpy as np
from vqregressor import vqregressor

ndata = 20
# random data
data = np.random.uniform(-1, 1, ndata)
# labeling them
labels = np.sin(2*data)

VQR = vqregressor(layers=1, data=data, labels=labels)
# set the training hyper-parameters
epochs = 50
learning_rate = 0.1

# perform the training
history = VQR.gradient_descent(learning_rate=learning_rate, epochs=epochs)
VQR.show_predictions('Predictions', save=True)

np.save("best_params", VQR.params)