
# import qibo's packages
import qibo
from qibo import gates, hamiltonians, derivative
from qibo.models import Circuit
from qibo.models.error_mitigation import CDR
from qibo.symbols import Z

# some useful python package
import numpy as np
import matplotlib.pyplot as plt

# numpy backend is enough for a 1-qubit model
qibo.set_backend('numpy')

class vqregressor:

  def __init__(self, data, labels, layers, nqubits=1):
    """Class constructor."""
    # some general features of the QML model
    self.nqubits = nqubits
    self.layers = layers
    self.data = data
    self.labels = labels

    # initialize the circuit and extract the number of parameters
    self.circuit = self.ansatz(nqubits, layers)

    # get the number of parameters
    self.nparams = nqubits * layers * 3
    # set the initial value of the variational parameters
    self.params = np.random.randn(self.nparams)
    # scaling factor for custom parameter shift rule
    self.scale_factors = np.ones(self.nparams)

# ---------------------------- ANSATZ ------------------------------------------

  def ansatz(self, nqubits, layers):
    """Here we implement the variational model ansatz."""
    c = Circuit(nqubits)
    for q in range(nqubits):
      for l in range(layers):
        # decomposition of RY gate
        c.add([
          gates.RX(q=q, theta=np.pi/2, trainable=False),
          gates.RZ(q=q, theta=0+np.pi),
          gates.RX(q=q, theta=np.pi/2, trainable=False),
          gates.RZ(q=q, theta=np.pi, trainable=False)
        ])
        c.add(gates.RZ(q=q, theta=0))
    c.add(gates.M(0))

    return c

# --------------------------- RE-UPLOADING -------------------------------------

  def inject_data(self, x):
    """Here we combine x and params in order to perform re-uploading."""
    params = []
    index = 0
    
    for q in range(self.nqubits):
      for l in range(self.layers):
        # embed X
        params.append(self.params[index] * x + self.params[index + 1])
        params.append(self.params[index + 2])
        # update scale factors 
        # equal to x only when x is involved
        self.scale_factors[index] = x
        # we have three parameters per layer
        index += 3

    # update circuit's parameters
    self.circuit.set_parameters(params)


  def set_parameters(self, new_params):
        """Function which sets the new parameters into the circuit"""
        self.params = new_params


# ------------------------------- PREDICTIONS ----------------------------------

  def one_prediction(self, x):
    """This function calculates one prediction with fixed x."""
    self.inject_data(x)
    prob = self.circuit(nshots=1000).probabilities(qubits=[0])
    return prob[0] - prob[1]


  def predict_sample(self):
    """This function returns all predictions."""
    predictions = []
    for x in self.data:
      predictions.append(self.one_prediction(x))

    return predictions


# ------------------------ PERFORMING GRADIENT DESCENT -------------------------

  def parameter_shift(self, parameter_index, x):
    """This function performs the PSR for one parameter"""

    original = self.params.copy()
    shifted = self.params.copy()

    shifted[parameter_index] += (np.pi / 2) / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    forward = self.one_prediction(x)

    shifted[parameter_index] -= np.pi / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    backward = self.one_prediction(x)

    self.params = original

    result = 0.5 * (forward - backward) * self.scale_factors[parameter_index]
    return result


  def circuit_derivative(self, x):
    """Derivatives of the expected value of the target observable with respect 
    to the variational parameters of the circuit are performed via parameter-shift
    rule (PSR)."""
    dcirc = np.zeros(self.nparams)   
    
    for par in range(self.nparams):
      # read qibo documentation for more information about this PSR implementation
      dcirc[par] = self.parameter_shift(par, x)
    
    return dcirc


  def evaluate_loss_gradients(self):
    """This function calculates the derivative of the loss function with respect
    to the variational parameters of the model."""

    # we need the derivative of the loss
    # nparams-long vector
    dloss = np.zeros(self.nparams)
    # we also keep track of the loss value
    loss = 0

    # cycle on all the sample
    for x, y in zip(self.data, self.labels):
      # calculate prediction
      prediction = self.one_prediction(x)
      # derivative of E[O] with respect all thetas
      dcirc = self.circuit_derivative(x)
      # calculate loss and dloss
      mse = (prediction - y)
      loss += mse**2
      dloss += 2 * mse * dcirc

    return dloss, loss/len(self.data)
  

  def gradient_descent(self, learning_rate, epochs):
    """This function performs a full gradient descent strategy."""

    # we want to keep track of the loss function
    loss_history = []

    # the gradient descent strategy
    for epoch in range(epochs):
      dloss, loss = self.evaluate_loss_gradients()
      loss_history.append(loss)
      self.params -= learning_rate * dloss
      print(f'Loss at epoch: {epoch + 1} ', loss)
    
    return loss_history


# ---------------------- PLOTTING FUNCTION -------------------------------------

  def show_predictions(self, title, save=False):
    """This function shows the obtained results through a scatter plot."""

    # calculate prediction
    predictions = self.predict_sample()

    # draw the results
    plt.figure(figsize=(12,8))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(self.data, self.labels, color='orange', alpha=0.6, label='Original', s=70, marker='o')
    plt.scatter(self.data, predictions, color='purple', alpha=0.6, label='Predictions', s=70, marker='o')

    plt.legend()

    # we save all the images during the training in order to see the evolution
    if save:
      plt.savefig(str(title)+'.png')
      plt.close()

    plt.show()
