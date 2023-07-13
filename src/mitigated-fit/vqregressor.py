import os, random

# some useful python package
import numpy as np
import matplotlib.pyplot as plt

# import qibo's packages
import qibo
from qibo import gates
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models import Circuit
from qibo.models import error_mitigation
from qibo.symbols import Z

from savedata_utils import get_training_type

from bp_utils import bound_pred, bound_grad

# numpy backend is enough for a 1-qubit model
# qibo.set_backend('qibolab', platform='tii1q_b1')

class vqregressor:

    def __init__(
        self,
        data,
        labels,
        layers,
        example,
        nqubits=1,  
        backend=None,
        noise_model=None,
        bp_bound=True,
        nshots=1000,
        expectation_from_samples=True,
        obs_hardware=False,
        mitigation={"step": False, "final": False, "method": None},
        mit_kwargs={},
        scaler=lambda x: x,
    ):
        """Class constructor."""
        # some general features of the QML model
        self.nqubits = nqubits
        self.layers = layers
        self.data = data
        # get data dimensionality
        self.ndim = len(np.atleast_1d(data[0]))

        if nqubits != self.ndim:
            raise ValueError(
                f"Please select a number of qubits equal to the data dimensionality, which is {self.ndim}"
            )

        self.labels = labels
        self.ndata = len(labels)
        self.backend = backend
        self.noise_model = noise_model
        self.bp_bound = bp_bound
        self.nshots = nshots
        self.exp_from_samples = expectation_from_samples
        self.obs_hardware = obs_hardware
        self.mitigation = mitigation
        self.mit_kwargs = mit_kwargs
        self.scaler = scaler
        self.example = example

        if backend is None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            self.backend = GlobalBackend()

        # initialize the circuit and extract the number of parameters
        self.circuit = self.ansatz(nqubits, layers)
        self.print_model()

        # get the number of parameters
        self.nparams = (nqubits * layers * 4) - 2 * nqubits
        # set the initial value of the variational parameters
        np.random.seed(1234)
        self.params = np.random.randn(self.nparams)
        print("Initial guess:", self.params)

        # scaling factor for custom parameter shift rule
        self.scale_factors = np.ones(self.nparams)

        if mitigation['method'] is not None:
            self.mitigation['method'] = getattr(error_mitigation, mitigation['method'])
            self.mit_params = None

        qibo.set_backend("numpy")
        

    # ---------------------------- ANSATZ ------------------------------------------

    def ansatz(self, nqubits, layers):
        """Here we implement the variational model ansatz."""
        c = Circuit(nqubits, density_matrix=True)

        for l in range(layers):
            for q in range(nqubits):
                c.add(gates.I(q))
                # decomposition of RY gate
                c.add(
                    [
                        gates.RX(q=q, theta=np.pi/2, trainable=False),
                        gates.RZ(q=q, theta=0),
                        gates.RZ(q=q, theta=np.pi, trainable=False),
                        gates.RX(q=q, theta=np.pi/2, trainable=False),
                        gates.RZ(q=q, theta=np.pi, trainable=False),
                    ]
                )
                # add RZ if this is not the last layer
                if l != self.layers - 1:
                    c.add(gates.RZ(q=q, theta=0))
            
            # add entangling layer between layers
            if (l != self.layers - 1) and (self.nqubits > 1):
                for q in range(0, nqubits-1, 1):
                    c.add(gates.CNOT(q0=q, q1=q+1))
                c.add(gates.CNOT(q0=nqubits-1, q1=0))

        for q in range(nqubits):
            c.add(gates.I(q))

        return c
    

    # --------------------------- RE-UPLOADING -------------------------------------

    def inject_data(self, x):
        """Here we combine x and params in order to perform re-uploading."""
        params = []
        index = 0

        # make it work also if x is 1d
        x = np.atleast_1d(x)

        for l in range(self.layers):
            for q in range(self.nqubits):
                # embed x
                params.append(
                    self.params[index] * self.scaler(x[q]) + self.params[index + 1]
                )
                # update scale factors

                # equal to x only when x is involved
                self.scale_factors[index] = self.scaler(x[q])

                # add RZ if this is not the last layer
                if l != self.layers - 1:
                    params.append(self.params[index + 2] * x[q] + self.params[index + 3])
                    self.scale_factors[index + 2] = x[q]
                    # we have four parameters per layer
                    index += 4

        # update circuit's parameters
        self.circuit.set_parameters(params)

    # --------------------------- PRINT MODEL SPECS --------------------------------

    def print_model(self):
        """Show circuit's specificities"""
        print("Circuit ansatz")
        print(self.circuit.draw())
        print("Circuit's specs")
        print(self.circuit.summary())

    # ------------------------------ MODIFY PARAMS ---------------------------------

    def set_parameters(self, new_params):
        """Function which sets the new parameters into the circuit"""
        self.params = new_params

    def get_parameters(self):
        """Functions which saves the current variational parameters"""
        return self.params

    # ------------------------------- PREDICTIONS ----------------------------------

    def epx_value(self):
        """Helper function to compute the final circuit and the observable to be measured"""
        circuit = self.circuit.copy(deep=True)
        if self.obs_hardware:
            circuit.add(gates.Z(*range(self.nqubits)))
            circuit += self.circuit.invert()
            circuit.add(gates.M(*range(self.nqubits)))
            observable = np.zeros((2**self.nqubits, 2**self.nqubits))
            observable[0, 0] = 1
            observable = Hamiltonian(self.nqubits, observable, backend=self.backend)
        else:
            circuit.add(gates.M(*range(self.nqubits)))
            observable = SymbolicHamiltonian(
                np.prod([Z(i) for i in range(self.nqubits)]), backend=self.backend
            )

        return circuit, observable

    def one_prediction(self, x):
        """This function calculates one prediction with fixed x."""
        self.inject_data(x)
        circuit, observable = self.epx_value()
        if self.noise_model != None:
            circuit = self.noise_model.apply(circuit)
        if self.exp_from_samples:
            obs = self.backend.execute_circuit(
                circuit, nshots=self.nshots
            ).expectation_from_samples(observable)
        else:
            obs = observable.expectation(
                self.backend.execute_circuit(circuit, nshots=self.nshots).state()
            )
        if self.obs_hardware:
            obs = np.sqrt(abs(obs))
        return obs

    def one_prediction_readout(self, x):
        """This function calculates one prediction with fixed x and readout mitigation."""
        self.inject_data(x)
        circuit, observable = self.epx_value()
        if self.noise_model != None:
            circuit = self.noise_model.apply(circuit)
        if self.exp_from_samples:
            result = self.backend.execute_circuit(circuit, nshots=self.nshots)
            readout_args = self.mit_kwargs['readout']
            if readout_args != {}:
                result = error_mitigation.apply_readout_mitigation(result, readout_args['calibration_matrix'])
            obs = result.expectation_from_samples(observable)
        else:
            obs = observable.expectation(self.backend.execute_circuit(circuit, nshots=self.nshots).state())
        if self.obs_hardware:
            obs = np.sqrt(abs(obs))
        return obs
   
    def get_fit(self):
        mean_params = []
        mit_kwargs = {key: self.mit_kwargs[key] for key in ['n_training_samples','readout']}
        cdr_data = []
        for _ in range(self.mit_kwargs['N_mean']):
            self.circuit.set_parameters(np.random.uniform(-np.pi,np.pi,int(self.nparams/2)))
            self.inject_data(self.data[random.randint(0,self.ndata-1)])
            circuit, observable = self.epx_value()
            cdr = self.mitigation['method'](
                circuit=circuit,
                observable=observable,
                noise_model=self.noise_model,
                backend=self.backend,
                nshots=self.nshots,
                full_output=True,
                **mit_kwargs
            )
            mean_params.append(cdr[2])
            print(cdr[2])
            cdr_data.append(cdr)
        mean_params = np.mean(mean_params,axis=0)
        return mean_params, cdr_data

    def one_mitigated_prediction(self, x):
        """This function calculates one mitigated prediction with fixed x."""
        obs_noisy = self.one_prediction_readout(x)
        if self.mit_params is None:
            self.mit_params = self.get_fit()[0]
        obs = self.mit_params[0]*obs_noisy + self.mit_params[1]
        return obs

    def step_prediction(self, x):
        if self.mitigation["step"]:
            prediction = self.one_mitigated_prediction
        else:
            prediction = self.one_prediction
        return prediction(x)

    def predict_sample(self):
        """This function returns all predictions."""
        if self.mitigation["final"]:
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

    # ------------------------- Derivative of <O> ----------------------------------

    def circuit_derivative(self, x):
        """Derivatives of the expected value of the target observable with respect
        to the variational parameters of the circuit are performed via parameter-shift
        rule (PSR)."""
        dcirc = np.zeros(self.nparams)
        
        for par in range(self.nparams):
            # read qibo documentation for more information about this PSR implementation
            dcirc[par] = self.parameter_shift(par, x)

        return dcirc

    # ---------------------- Derivative of the loss function -----------------------

    def evaluate_loss_gradients(self, data=None, labels=None):
        """This function calculates the derivative of the loss function with respect
        to the variational parameters of the model."""

        if self.noise_model is not None:
            params = self.noise_model.errors[gates.I][0][1].options
            probs = [params[k][1] for k in range(4**self.nqubits-1)]
            bit_flip = self.noise_model.errors[gates.M][0][1].options[0,-1]**(1/self.nqubits)
        else:
            probs = np.zeros(4**self.nqubits-1)
            bit_flip = 0

        if self.bp_bound:
            bound_grads = bound_grad(self.layers, self.nqubits, probs, bit_flip)
            bound_preds = bound_pred(self.layers, self.nqubits, probs, bit_flip)

        if data is None:
            data = self.data

        if labels is None:
            labels = self.labels

        # we need the derivative of the loss
        # nparams-long vector
        dloss = np.zeros(self.nparams)
        dloss_bound = 0
        # we also keep track of the loss value
        loss = 0
        loss_bound = 0
        # cycle on all the sample
        for x, y in zip(data, labels):
            # calculate prediction
            prediction = self.step_prediction(x)
            # derivative of E[O] with respect all thetas
            dcirc = self.circuit_derivative(x)
            # calculate loss and dloss
            mse = prediction - y
            loss += mse**2
            dloss += 2 * mse * dcirc
            if self.bp_bound:
                dloss_bound += (2 * mse * bound_grads)**2
                if y - bound_preds > 0:
                    loss_bound += (y - bound_preds)**2
        

        return dloss / len(data), loss / len(data), np.sqrt(dloss_bound) / len(data), loss_bound / len(data)

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

        grads, loss, dloss_bound, loss_bound  = self.evaluate_loss_gradients(data, labels)
        
        m = beta_1 * m + (1 - beta_1) * grads
        v = beta_2 * v + (1 - beta_2) * grads * grads
        mhat = m / (1.0 - beta_1 ** (iteration + 1))
        vhat = v / (1.0 - beta_2 ** (iteration + 1))
        self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

        return m, v, loss, grads, dloss_bound, loss_bound

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
        return iter(
            zip(np.array_split(data, nbatches), np.array_split(labels, nbatches))
        )

    # ---------------------- Gradient Descent ------------------------------------
    def gradient_descent(
        self,
        learning_rate,
        epochs,
        batchsize=10,
        restart_from_epoch=None,
        method="Adam",
        J_treshold=1e-11,
        live_plotting=True,
    ):
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

        if method != "Adam" and method != "Standard":
            raise ValueError(
                print(
                    "This method does not exist. Please select one of the following: Adam, Standard."
                )
            )

        cache_dir = f"{self.example}/cache"

        if self.noise_model is not None:
            noise = True
        else:
            noise = False
        train_type = get_training_type(self.mitigation, noise)

        # creating folder where to save params during training
        if not os.path.exists(f"{cache_dir}/params_history_{train_type}"):
            os.makedirs(f"{cache_dir}/params_history_{train_type}")

        # resuming old training
        if restart_from_epoch is not None:
            print(f"Resuming parameters from epoch {restart_from_epoch}")
            resume_params = np.load(
                f"{cache_dir}/params_history_{train_type}/params_epoch_{restart_from_epoch}.npy"
            )
            self.set_parameters(resume_params)
        else:
            restart_from_epoch = 0

        if restart_from_epoch is None:
            restart = 0
        else:
            restart = restart_from_epoch

        # we track the loss history
        loss_history, grad_history, grad_bound_history, loss_bound_history = [], [], [], []
        
        if self.mitigation['step'] is True:
            cdr_history = []

        # useful if we use adam optimization
        if method == "Adam":
            m = np.zeros(self.nparams)
            v = np.zeros(self.nparams)

        # cycle over the epochs
        for epoch in range(epochs):

            if self.mitigation['step'] is True and epoch%self.mit_kwargs['N_update']==0:   
                self.mit_params, cdr_data = self.get_fit()
                cdr_history.append(cdr_data)

            iteration = 0

            # stop the training if the target loss is reached
            if False:
                if epoch != 0 and loss_history[-1] < J_treshold:
                    print(
                        "Desired sensibility is reached, here we stop: ",
                        iteration,
                        " iteration",
                    )
                    break

            # run over the batches
            for data, labels in self.data_loader(batchsize):
                # update iteration tracker
                iteration += 1

                # update parameters using the chosen method
                if method == "Adam":
                    m, v, loss, grads, dloss_bound, loss_bound = self.apply_adam(
                        learning_rate, m, v, data, labels, iteration
                    )
                elif method == "Standard":
                    grads, loss = self.evaluate_loss_gradients()
                    self.params -= learning_rate * dloss
                
                grad_history.append(grads)
                loss_history.append(loss)
                if self.bp_bound:
                    grad_bound_history.append(dloss_bound)
                    loss_bound_history.append(loss_bound)

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
                    self.show_predictions(f"Live_predictions", save=True)

            np.save(
                arr=self.params,
                file=f"{cache_dir}/params_history_{train_type}/params_epoch_{epoch + restart + 1}",
            )

            
        name = ""
        if self.noise_model is not None:
            name += "noisy"
        if self.mitigation['method'] is not None:
            name += f"_{self.mitigation['method'].__name__}"
            if self.mitigation['step']:
                name += "-step"
            if self.mitigation['final']:
                name += "-final"
            if self.mitigation['readout']:
                name += f"-readout"
        # if self.noise_model is not None:
        #     noise = True
        # else:
        #     noise = False
        # train_type = get_training_type(self.mitigation, noise)
        np.save(arr=np.asarray(loss_history), file=f"{cache_dir}/loss_history_{train_type}")
        np.save(arr=np.asarray(grad_history), file=f"{cache_dir}/grad_history_{train_type}")
        if self.bp_bound:
            np.save(arr=np.asarray(grad_bound_history), file=f"{cache_dir}/grad_bound_history_{train_type}")
            np.save(arr=np.asarray(loss_bound_history), file=f"{cache_dir}/loss_bound_history_{train_type}")
        if self.mitigation['step'] is True:
            np.save(arr=np.asanyarray(cdr_data, dtype=object), file=f"{cache_dir}/cdr_history_{train_type}")

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
            loss += (prediction - label) ** 2

        return loss / self.ndata

    # ---------------------------- CMA OPTIMIZATION --------------------------------
    def cma_optimization(self):
        """Method which performs a GA optimization."""

        myloss = self.loss
        # this can be used to stop the optimization once reached a target J value
        # it must be added as argument of cma.fmin2 by typing options=options
        options = {"ftarget": 5e-5}
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

        if self.ndim != 1:
            x_0_array = self.data.T[0]
        else:
            x_0_array = self.data

        if self.noise_model is not None:
            params = self.noise_model.errors[gates.I][0][1].options
            probs = [params[k][1] for k in range(4**self.nqubits-1)]
            bit_flip = self.noise_model.errors[gates.M][0][1].options[0,-1]**(1/self.nqubits)
        else:
            probs = np.zeros(4**self.nqubits-1)
            bit_flip = 0

        if self.bp_bound:
            bounds = bound_pred(self.layers, self.nqubits, probs, bit_flip)

        # draw the results
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(
            x_0_array,
            self.labels,
            color="orange",
            alpha=0.6,
            label="Original",
            s=70,
            marker="o",
        )
        plt.scatter(
            x_0_array,
            predictions,
            color="purple",
            alpha=0.6,
            label="Predictions",
            s=70,
            marker="o",
        )

        if self.bp_bound:
            plt.plot(
                x_0_array,
                [bounds]*len(x_0_array),
                color="black",
                alpha=0.6,
                label="BP Bound",
            )

        plt.legend()

        # we save all the images during the training in order to see the evolution
        if save:
            plt.savefig(str(title) + ".png")
            plt.close()

        plt.show()
