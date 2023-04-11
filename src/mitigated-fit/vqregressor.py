
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

  def __init__(self, data, labels, layers, nqubits=1, backend=None, noise_model=None, nshots=1000, expectation_from_samples=True, obs_hardware=False, mitigation={'step':False,'final':False,'method':None}, mit_kwargs={}, scaler=lambda x: x):
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
    self.exp_from_samples = expectation_from_samples
    self.obs_hardware = obs_hardware
    self.mitigation = mitigation
    self.mit_kwargs = mit_kwargs
    self.scaler = scaler
    # default observable. It will be modified if obs_hardware == True
    self.observable = SymbolicHamiltonian(np.prod([ Z(i) for i in range(self.nqubits) ]))

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend

      self.backend = GlobalBackend()
    if mitigation['method'] is not None:
      self.mitigation['method'] = getattr(error_mitigation, mitigation['method'])

    # initialize the circuit and extract the number of parameters
    self.circuit = self.ansatz(nqubits, layers)

    # set the number of parameters
    self.real_nparams = (nqubits * layers * 4) - 2

    if obs_hardware:  
      self.nparams = 2 * self.real_nparams
      parameters_set = np.random.randn(self.real_nparams)
      # original params concatenated with the reverse array
      # because circ will be e.g.: gate1-gate2-OBS-gate2-gate1
      self.params = np.concatenate((parameters_set, parameters_set[::-1]))
    else:
      self.nparams = self.real_nparams
      self.params = np.random.randn(self.nparams)

    # set the initial value of the variational parameters
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
      
    if self.obs_hardware:

      inv_c = c.invert()
      c.add(gates.Z(*range(self.nqubits)))
      c += inv_c
      c.add(gates.M(*range(self.nqubits)))
      observable = np.zeros((2**self.nqubits,2**self.nqubits))
      observable[0,0] = 1
      self.observable = Hamiltonian(self.nqubits, observable)
    else:
      # we only add measurements because observable is already in the right form
      c.add(gates.M(*range(self.nqubits)))

    print(c.draw())

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
        # updating also for the inverse circuit
        if self.obs_hardware:
          self.scale_factors[self.nparams - index - 1] = self.scaler(x)

        # add RZ if this is not the last layer
        if(l != self.layers - 1):
          params.append(self.params[index + 2] * x + self.params[index + 3])
          self.scale_factors[index + 2] = x

          if self.obs_hardware:
            # nparams - (index + 2) - 1
            self.scale_factors[self.nparams - index - 3] = x

          index += 4
      
      # filling also params of the inverse circuit
      if self.obs_hardware:
        params = np.concatenate((params, params[::-1]))

    # update circuit's parameters
    self.circuit.set_parameters(params)


  def set_parameters(self, new_params):
        """Function which sets the new parameters into the circuit"""
        if self.obs_hardware:
          params = np.concatenate((new_params, new_params[::-1]))
        else:
          params = new_params
        
        self.params = params

  
  def get_parameters(self):
    """Functions which saves the current variational parameters"""
    return self.params


# ------------------------------- PREDICTIONS ----------------------------------


  def one_prediction(self, x):
    """This function calculates one prediction with fixed x."""
    self.inject_data(x)
    if self.noise_model != None:
      circuit = self.noise_model.apply(self.circuit)
    if self.exp_from_samples:
      obs = self.backend.execute_circuit(self.circuit, nshots=self.nshots).expectation_from_samples(self.observable)
    else:
      obs = self.observable.expectation(self.backend.execute_circuit(self.circuit, nshots=self.nshots).state())
    if self.obs_hardware:
        obs = np.sqrt(abs(obs))
    return obs


  def one_mitigated_prediction(self, x):
    """This function calculates one mitigated prediction with fixed x."""
    self.inject_data(x)
    obs = self.mitigation['method'](
      circuit=self.circuit,
      observable=self.observable,
      noise_model=self.noise_model,
      backend=self.backend,
      nshots=self.nshots,
      **self.mit_kwargs
    )
    if self.obs_hardware:
      obs = np.sqrt(abs(obs))
    return obs


  def step_prediction(self, x):
    if self.mitigation['step']:
      prediction = self.one_mitigated_prediction
    else:
      prediction = self.one_prediction
    return prediction(x)
  

  def predict_sample(self):
    """This function returns all predictions."""
    if self.mitigation['final']:
      prediction = self.one_mitigated_prediction
    else:
      prediction = self.one_prediction
    predictions = []
    for x in self.data:
      predictions.append(prediction(x))

    return predictions


# ------------------------ PERFORMING GRADIENT DESCENT -------------------------
# --------------------------- Parameter Shift Rule -----------------------------

  def parameter_shift(self, parameter_index, x):
    """This function performs the PSR for one parameter"""

    original = self.params.copy()
    shifted = self.params.copy()

    shifted[parameter_index] += (np.pi / 2) / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    forward = self.step_prediction(x)

    shifted[parameter_index] -= np.pi / self.scale_factors[parameter_index]
    self.set_parameters(shifted)
    backward = self.step_prediction(x)

    self.params = original

    result = 0.5 * (forward - backward) * self.scale_factors[parameter_index]
    return result
  

# ------------------------- gradient check -------------------------------------

  def tf_gradients(self, x):
    """Checks gradients calculated using tensorflow gradient tape."""

    qibo.set_backend('tensorflow')
    import tensorflow as tf

    with tf.GradientTape as tape:
      tape.watch(self.params)
      prediction = self.step_prediction(x)

    return tape.gradient(prediction, [self.params])

# ------------------------- Derivative of <O> ----------------------------------

  def circuit_derivative(self, x):
    """Derivatives of the expected value of the target observable with respect 
    to the variational parameters of the circuit are performed via parameter-shift
    rule (PSR).
    It returns a real_nparams long vector of derivatives."""
    dcirc = np.zeros(self.real_nparams)   
    
    for par in range(self.real_nparams):
      dcirc[par] = self.parameter_shift(par, x)
      # adding contribution from the second parameter
      if self.obs_hardware:
        dcirc[par] += self.parameter_shift(self.nparams-par-1, x)

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
    dloss = np.zeros(self.real_nparams)
    # we also keep track of the loss value
    loss = 0

    # cycle on all the sample
    for x, y in zip(data, labels):
      # calculate prediction
      prediction = self.step_prediction(x)
      # derivative of E[O] with respect all thetas
      dcirc = self.circuit_derivative(x)
      # calculate loss and dloss
      mse = (prediction - y)
      loss += mse**2
      dloss += 2 * mse * dcirc

    if self.obs_hardware:
      dloss = np.concatenate((dloss, dloss[::-1]))

    return dloss, loss/len(data)

    
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

    m = beta_1 * m + (1 - beta_1) * grads
    v = beta_2 * v + (1 - beta_2) * grads * grads
    mhat = m / (1.0 - beta_1 ** (iteration + 1))
    vhat = v / (1.0 - beta_2 ** (iteration + 1))
    self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

    return m, v, loss


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
          self.params -= learning_rate * dloss

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
      prediction = self.step_prediction(x)
      loss += (prediction -  label)**2
    
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
      plt.savefig(str(title) + '.png')
      plt.close()

    plt.show()
