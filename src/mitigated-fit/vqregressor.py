
# import qibo's packages
import qibo
from qibo import gates
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models import Circuit
from qibo.models import error_mitigation
from qibo.symbols import Z

# some useful python package
import numpy as np
import matplotlib.pyplot as plt

# numpy backend is enough for a 1-qubit model
qibo.set_backend('numpy')

class vqregressor:

  def __init__(self, data, labels, layers, nqubits=1, backend=None, noise_model=None, min_shots=10, nshots=1000, expectation_from_samples=True, obs_hardware=False, mitigation={'step':False,'final':False,'method':None}, mit_kwargs={}, scaler=lambda x: x):
    """Class constructor."""
    # some general features of the QML model
    self.nqubits = nqubits
    self.layers = layers
    self.data = data
    self.labels = labels
    self.ndata = len(labels)
    self.backend = backend
    self.noise_model = noise_model
    self.nshots = nshots
    self.min_shots = min_shots
    self.exp_from_samples = expectation_from_samples
    self.obs_hardware = obs_hardware
    self.mitigation = mitigation
    self.mit_kwargs = mit_kwargs
    self.scaler = scaler

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend

      self.backend = GlobalBackend()
    if mitigation['method'] is not None:
      self.mitigation['method'] = getattr(error_mitigation, mitigation['method'])

    # initialize the circuit and extract the number of parameters
    self.circuit = self.ansatz(nqubits, layers)

    # get the number of parameters
    self.nparams = (nqubits * layers * 4) - 2
    self.nshots_param = np.zeros(self.nparams) + min_shots
    # set the initial value of the variational parameters
    self.params = np.random.randn(self.nparams)
    # scaling factor for custom parameter shift rule
    self.scale_factors = np.ones(self.nparams)

# ---------------------------- ANSATZ ------------------------------------------

  def ansatz(self, nqubits, layers):
    """Here we implement the variational model ansatz."""
    c = Circuit(nqubits, density_matrix=True)
    for q in range(nqubits):
      for l in range(layers):
        # decomposition of RY gate
        c.add([
          gates.RX(q=q, theta=np.pi/2, trainable=False),
          gates.RZ(q=q, theta=0),
          gates.RZ(q=q, theta=np.pi, trainable=False),
          gates.RX(q=q, theta=np.pi/2, trainable=False),
          gates.RZ(q=q, theta=np.pi, trainable=False)
        ])
        # add RZ if this is not the last layer
        if(l != self.layers - 1):
          c.add(gates.RZ(q=q, theta=0))

    return c

# --------------------------- RE-UPLOADING -------------------------------------

  def inject_data(self, x):
    """Here we combine x and params in order to perform re-uploading."""
    params = []
    index = 0
    
    for q in range(self.nqubits):
      for l in range(self.layers):
        # embed x
        params.append(self.params[index] * self.scaler(x) + self.params[index + 1])
        # update scale factors 

        # equal to x only when x is involved
        self.scale_factors[index] = self.scaler(x)

        # add RZ if this is not the last layer
        if(l != self.layers - 1):
          params.append(self.params[index + 2] * x + self.params[index + 3])
          self.scale_factors[index + 2] = x
          # we have four parameters per layer
          index += 4

    # update circuit's parameters
    self.circuit.set_parameters(params)


  def set_parameters(self, new_params):
        """Function which sets the new parameters into the circuit"""
        self.params = new_params

  
  def get_parameters(self):
    """Functions which saves the current variational parameters"""
    return self.params


# ------------------------------- PREDICTIONS ----------------------------------

  def epx_value(self):
    """Helper function to compute the final circuit and the observable to be measured"""
    circuit = self.circuit.copy(deep = True)
    if self.obs_hardware:
      circuit.add(gates.Z(*range(self.nqubits)))
      circuit += self.circuit.invert()
      circuit.add(gates.M(*range(self.nqubits)))
      observable = np.zeros((2**self.nqubits,2**self.nqubits))
      observable[0,0] = 1
      observable = Hamiltonian(self.nqubits, observable)
    else:
      circuit.add(gates.M(*range(self.nqubits)))
      observable = SymbolicHamiltonian(np.prod([ Z(i) for i in range(self.nqubits) ]))

    return circuit, observable
  
  
  def expectation_from_samples(obs, result):
    from collections import Counter
    samples = result.samples()
    exp = []
    for j in samples:
        freq = Counter()
        freq[str(j[0])] = 1
        exp.append(obs.expectation_from_samples(freq))
    var = np.var(exp)
    exp = np.mean(exp)
    return exp, var # In our case obs = Z, so Var(Z) = <Z^2> - <Z>^2 = 1 - <Z>^2 = 1 - exp^2. If obs_hardware==True, obs=Projector, so Var(obs) = <obs> - <obs>^2
  
  
  def one_prediction(self, x, nshots):
    """This function calculates one prediction with fixed x."""
    self.inject_data(x)
    circuit, observable = self.epx_value()
    if self.noise_model != None:
      circuit = self.noise_model.apply(circuit)
    if self.exp_from_samples:
      result = self.backend.execute_circuit(circuit, nshots=nshots)
      obs, var = self.expectation_from_samples(observable, result)
      # var = 1 - obs**2 (in this case)
    else:
      obs = observable.expectation(self.backend.execute_circuit(circuit, nshots=self.nshots).state())
      var = 0
    if self.obs_hardware:
        obs = np.sqrt(abs(obs))
        # variance does not change
    return obs, var


  def one_mitigated_prediction(self, x, nshots):
    """This function calculates one mitigated prediction with fixed x."""
    self.inject_data(x)
    circuit, observable = self.epx_value()
    obs = self.mitigation['method'](
      circuit=circuit,
      observable=observable,
      noise_model=self.noise_model,
      backend=self.backend,
      nshots=nshots,
      **self.mit_kwargs
    )
    var = 1 - obs**2
    if self.obs_hardware:
      obs = np.sqrt(abs(obs))
      # variance does not change
    return obs, var


  def step_prediction(self, x, nshots):
    if self.mitigation['step']:
      prediction = self.one_mitigated_prediction
    else:
      prediction = self.one_prediction
    return prediction(x, nshots)
  

  def predict_sample(self):
    """This function returns all predictions."""
    if self.mitigation['final']:
      prediction = self.one_mitigated_prediction
    else:
      prediction = self.one_prediction
    predictions = []
    for x in self.data:
      predictions.append([prediction(x, self.nshots)])

    return predictions


# ------------------------ PERFORMING GRADIENT DESCENT -------------------------
# --------------------------- Parameter Shift Rule -----------------------------

  def parameter_shift(self, parameter_index, x):
    """This function performs the PSR for one parameter"""

    original = self.params.copy()
    shifted = self.params.copy()
    nshots = self.nshots_param[parameter_index]
    shifted[parameter_index] += (np.pi / 2) / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    forward = np.array(self.step_prediction(x, nshots))

    shifted[parameter_index] -= np.pi / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    backward = np.array(self.step_prediction(x, nshots))

    self.params = original

    result = 0.5 * (forward[:,0] - backward[:,0]) * self.scale_factors[parameter_index]
    var = 0.25 * (forward[:,1] - backward[:,1]) * self.scale_factors[parameter_index]
    return result, var

# ------------------------- Derivative of <O> ----------------------------------

  def circuit_derivative(self, x):
    """Derivatives of the expected value of the target observable with respect 
    to the variational parameters of the circuit are performed via parameter-shift
    rule (PSR)."""
    dcirc = np.zeros(self.nparams, 2)   
    
    for par in range(self.nparams):
      # read qibo documentation for more information about this PSR implementation
      dcirc[par, :] = self.parameter_shift(par, x) #0 -> grad, 1-> variance
    
    return dcirc

  
# ---------------------- Derivative of the loss function -----------------------


  def evaluate_loss_gradients(self, data=None, labels=None):
    """This function calculates the derivative of the loss function with respect
    to the variational parameters of the model."""

    if data is None:
      data = self.data
    
    if labels is None:
      labels = self.labels

    # we need the derivative of the loss
    # nparams-long vector
    dloss = np.zeros(self.nparams, 2)
    # we also keep track of the loss value
    loss = 0

    # cycle on all the sample
    for x, y in zip(data, labels):
      # calculate prediction
      prediction = np.array(self.step_prediction(x))
      # derivative of E[O] with respect all thetas
      dcirc = self.circuit_derivative(x)
      # calculate loss and dloss
      mse = (prediction[:,0] - y)
      loss += mse**2
      dloss[:, 0] += 2 * mse * dcirc[:, 0]
      dloss[:, 1] += 4 * mse**2 * dcirc[:, 1] #variance
    
    loss /= len(data)
    dloss[:, 0] /= len(data)
    dloss[:, 1] /= len(data)**2

    return dloss, loss

    
# ---------------------- Update parameters if we use Adam ----------------------


  def apply_adam(
    self,
    learning_rate,
    m,
    v,
    data,
    labels,
    iteration,
    beta_1=0.85,
    beta_2=0.99,
    epsilon=1e-8,
  ):
    """
    Implementation of the Adam optimizer: during a run of this function parameters are updated.
    Furthermore, new values of m and v are calculated.
    Args:
        learning_rate: np.float value of the learning rate
        m: momentum's value before the execution of the Adam descent
        v: velocity's value before the execution of the Adam descent
        features: np.matrix containig the n_sample-long vector of states
        labels: np.array of the labels related to features
        iteration: np.integer value corresponding to the current training iteration
        beta_1: np.float value of the Adam's beta_1 parameter; default 0.85
        beta_2: np.float value of the Adam's beta_2 parameter; default 0.99
        epsilon: np.float value of the Adam's epsilon parameter; default 1e-8
    Returns: np.float new values of momentum and velocity
    """

    grads, loss = self.evaluate_loss_gradients(data, labels)
    grads = grads[:,0]

    m = beta_1 * m + (1 - beta_1) * grads
    v = beta_2 * v + (1 - beta_2) * grads * grads
    mhat = m / (1.0 - beta_1 ** (iteration + 1))
    vhat = v / (1.0 - beta_2 ** (iteration + 1))
    self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

    return m, v, loss
  
  def apply_rosalin(
      self,
      learning_rate,
      shots_used,
      nshots_param,
      lipschitz,
      b,
      mu,
      chi1,
      xi1,
      data,
      labels,
      iteration,
      ):
    
    grads, loss = self.evaluate_loss_gradients(data, labels)
    self.params -= learning_rate*grads[:,0]

    xi1 = mu * xi1 + (1 - mu) * grads[:,1]
    xi2 = xi1 / (1 - mu ** (iteration + 1))
    chi1 = mu * chi1 + (1 - mu) * grads[:,1]
    chi2 = chi1 / (1 - mu ** (iteration + 1))

    s = np.ceil(
        (2 * lipschitz * learning_rate * xi2)
        / ((2 - lipschitz * learning_rate) * (chi2 ** 2 + b * (mu ** iteration)))
    )
    
    gamma = (
        (learning_rate - lipschitz * learning_rate ** 2 / 2) * chi2 ** 2
        - xi2 * lipschitz * learning_rate ** 2 / (2 * s)
    ) / s

    argmax_gamma = np.unravel_index(np.argmax(gamma), gamma.shape)
    smax = s[argmax_gamma]
    nshots_param = np.clip(s, min(2, self.min_shots), smax)

    return nshots_param, chi1, xi1
    
    



  def data_loader(self, batchsize):
    """Returns a random batch of data with their labels"""
    # calculating number of batches if batchsize is chosen
    nbatches = int(self.ndata / batchsize) + (self.ndata % batchsize > 0)
    # all data indices 
    ind = np.arange(self.ndata)
    # permutating indices and so data and labels
    np.random.shuffle(ind)
    data = self.data[ind]
    labels = self.labels[ind]
    # returning data splitted into batches
    return iter(zip(
      np.array_split(data, nbatches),
      np.array_split(labels, nbatches)
    ))
  
  # ---------------------- Gradient Descent ------------------------------------

  def gradient_descent(self, 
    learning_rate, 
    epochs, 
    batchsize = 10,
    restart_from_epoch=None, 
    method='Adam',
    J_treshold = 1e-5,
    live_plotting=True):

    """
    This function performs a full gradient descent strategy.
    
    Args:
      learning_rate (float): learning rate.
      epochs (int): number of optimization epochs.
      batches (int): number of batches in which you want to split the training set.
      (default 1)
      restart_from_epoch (int): epoch from which you want to restart a previous 
      training (default None)
      method (str): gradient descent method you want to perform. Only "Standard"
      and "Adam" are available (default "Adam").
      J_treshold (float): target value for the loss function.
    """ 
     
    if(method != 'Adam' and method != 'Standard'):
      raise ValueError(
        print('This method does not exist. Please select one of the following: Adam, Standard.')
      )

    # resuming old training
    if restart_from_epoch is not None:
      resume_params = np.load(f"results/params_psr/params_epoch_{restart_from_epoch}.npy")
      self.set_parameters(resume_params)
    else:
      restart_from_epoch = 0

    # we track the loss history
    loss_history = []

    # useful if we use adam optimization
    if(method == 'Adam'):
      m = np.zeros(self.nparams)
      v = np.zeros(self.nparams)

    # cycle over the epochs
    for epoch in range(epochs):
      
      iteration = 0
      
      # stop the training if the target loss is reached
      if(epoch != 0 and loss_history[-1] < J_treshold):
        print(
          "Desired sensibility is reached, here we stop: ",
          iteration,
          " iteration"
        )
        break

      # run over the batches
      for data, labels in self.data_loader(batchsize):
        # update iteration tracker
        iteration += 1

        # update parameters using the chosen method
        if(method=='Adam'):
          m, v, loss = self.apply_adam(
              learning_rate, m, v, data, labels, iteration
          )
        elif(method=='Standard'):
          dloss, loss = self.evaluate_loss_gradients()
          self.params -= learning_rate * dloss[:, 0]

        loss_history.append(loss)

        # track the training
        print(
            "Iteration ",
            iteration,
            " epoch ",
            epoch + 1,
            " | loss: ",
            loss,
        )

        if live_plotting:
          self.show_predictions(f'Live_predictions', save=True)
    
    return loss_history



# ------------------------ LOSS FUNCTION ---------------------------------------

  
  def loss(self, params=None):
    """This function calculates the loss function for the entire sample."""

    # it can be useful to pass parameters as argument when we perform the cma
    if params is None:
      params = self.params

    loss = 0
    self.set_parameters(params)

    for x, label in zip(self.data, self.labels):
      prediction = self.step_prediction
      loss += (prediction[:, 0] -  label)**2
    
    return loss/self.ndata



# ---------------------------- CMA OPTIMIZATION --------------------------------

  def cma_optimization(self):
      """Method which performs a GA optimization."""
      
      myloss = self.loss
      # this can be used to stop the optimization once reached a target J value
      # it must be added as argument of cma.fmin2 by typing options=options
      options = {'ftarget':5e-5}
      import cma

      r = cma.fmin2(lambda p: myloss(p), self.params, 2, options=options)
      result = r[1].result.fbest
      parameters = r[1].result.xbest
      
      return result, parameters


# ---------------------- PLOTTING FUNCTION -------------------------------------

  def show_predictions(self, title, save=False):
    """This function shows the obtained results through a scatter plot."""

    # calculate prediction
    predictions = np.array(self.predict_sample())[:,0]

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
      plt.savefig(str(title) + '.png')
      plt.close()

    plt.show()
