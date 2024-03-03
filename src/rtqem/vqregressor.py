import os, random
from joblib import Parallel, delayed

# some useful python package
import numpy as np
import matplotlib.pyplot as plt

# qibo's
import qibo
from qibo import gates
from qibo.config import log
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models import Circuit
from qibo.models import error_mitigation
from qibo.models.error_mitigation import error_sensitive_circuit, apply_resp_mat_readout_mitigation
from qibo.symbols import Z
from utils import fuse
from qibo.backends import GlobalBackend

# rtqem's
from savedata_utils import get_training_type
from bp_utils import bound_pred, bound_grad, generate_noise_model

class VQRegressor:

    def __init__(
        self,
        data,
        labels,
        layers,
        example,
        nqubits=1,
        qubit_map = None,
        backend=None,
        nthreads=1,
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

        if qubit_map == None:
            qubit_map = list(range(self.nqubits))

        self.qubit_map = qubit_map
        self.noise_model = noise_model[0]
        self.noise_update = noise_model[1]
        self.noise_threshold = noise_model[2]
        self.evolution_model = noise_model[3]
        self.evolution_parameter = noise_model[4]
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

        self.nthreads = nthreads
        # initialize the circuit and extract the number of parameters
        self.circuit = self.ansatz(nqubits, layers)
        self.print_model()

        # get the number of parameters
        self.nparams = (nqubits * layers * 4) #- 2 * nqubits
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
            #c.add(gates.I(*range(nqubits)))
            for q in range(nqubits):
                Id = gates.I(q)
                if l == 0:
                    Id.name = 'id_init'
                c.add(Id)
                # decomposition of RY gate
                gpi2 = gates.GPI2(q=q, phi=0, trainable=False)
                gpi2.clifford = True
                c.add(
                    [
                        gpi2,
                        gates.RZ(q=q, theta=0.5),
                        gates.RZ(q=q, theta=np.pi, trainable=False),
                        gpi2,
                        gates.RZ(q=q, theta=np.pi, trainable=False),
                    ]
                )
                # add RZ if this is not the last layer
                #if l != self.layers - 1:
                c.add(gates.RZ(q=q, theta=0.5))

            # add entangling layer between layers
            #if (l != self.layers - 1) and (self.nqubits > 1):
            if (self.nqubits > 1):
                for q in range(0, nqubits-1, 1):
                    c.add(gates.CNOT(q0=q, q1=q+1))
                c.add(gates.CNOT(q0=nqubits-1, q1=0))
        #c.add(gates.I(*range(nqubits)))
        for q in range(nqubits):
            Id = gates.I(q)
            Id.name = 'id_end'
            c.add(Id)

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
                #if l != self.layers - 1:
                params.append(self.params[index + 2] * x[q] + self.params[index + 3])
                self.scale_factors[index + 2] = x[q]
                # we have four parameters per layer
                index += 4
                # TODO: reduce the fluctuations adding: self.scaler(x[q])/10
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

    def transpile_circ(self, circuit):
        from qibolab.transpilers.unitary_decompositions import u3_decomposition
        if self.backend.name == 'qibolab':
            new_c = Circuit(self.backend.platform.nqubits, density_matrix=True)
            for gate in circuit.queue:
                qubits = [self.qubit_map[j] for j in gate.qubits]
                if isinstance(gate, gates.M):
                    new_gate = gates.M(*tuple(qubits), **gate.init_kwargs)
                    new_gate.result = gate.result
                    new_c.add(new_gate)
                elif isinstance(gate, gates.I):
                    new_c.add(gate.__class__(*tuple(qubits), **gate.init_kwargs))
                else:
                    matrix = gate.matrix()
                    new_c.add(gates.U3(*tuple(qubits), *u3_decomposition(matrix)))
            return new_c
        else:
            return circuit

    # ------------------------------- PREDICTIONS ----------------------------------

    def epx_value(self):
        """Helper function to compute the final circuit and the observable to be measured"""
        circuit = self.circuit.copy(deep=True)
        if self.obs_hardware:
            circuit.add(gates.Z(*range(self.nqubits)))
            circuit += self.circuit.invert()

            circuit.add(gates.M(k) for k in range(self.nqubits))
            observable = np.zeros((2**self.nqubits, 2**self.nqubits))
            observable[0, 0] = 1
            observable = Hamiltonian(self.nqubits, observable, backend=self.backend)
        else:
            circuit.add(gates.M(k) for k in range(self.nqubits))
            observable = SymbolicHamiltonian(
                np.prod([Z(i) for i in range(self.nqubits)]), backend=self.backend
            )

        return circuit, observable

    def one_prediction(self, data, params=None):
        """This function calculates one prediction with fixed x."""
        circuits = []
        if params is None:
            params = [[self.params]]*len(data)
        for ii, x in enumerate(data):
            params1 = params[ii]
            for param in params1:
                self.params = param
                self.inject_data(x)
                circuit, observable = self.epx_value()
                if self.backend.name != 'QuantumSpain':
                    circuit = fuse(circuit, max_qubits=1)
                if self.noise_model != None:
                    circuit = self.noise_model.apply(circuit)
                if self.exp_from_samples and self.backend.name != 'QuantumSpain':
                    circuit = self.transpile_circ(circuit)
                circuits.append(circuit)

        if self.exp_from_samples:
            if self.backend.name == 'QuantumSpain':
                results = self.backend.execute_circuit(circuits, nshots=self.nshots)
            else:
                results = [self.backend.execute_circuit(circuit, nshots=self.nshots) for circuit in circuits]
        else:
            results = [self.backend.execute_circuit(circuit, nshots=self.nshots) for circuit in circuits]

        obs_list = []
        for result in results:
            if self.exp_from_samples:
                obs = observable.expectation_from_samples(result.frequencies())
            else:
                obs = observable.expectation(
                    result.state()
                )
            if self.obs_hardware:
                obs = np.sqrt(abs(obs))
            obs_list.append(obs)
        return obs_list

    def one_prediction_readout(self, data, params=None):
        """This function calculates one prediction with fixed x."""
        circuits = []
        if params is None:
            params = [[self.params]]*len(data)
        for ii, x in enumerate(data):
            params1 = params[ii]
            for param in params1:
                self.params = param
                self.inject_data(x)
                circuit, observable = self.epx_value()
                if self.backend.name != 'QuantumSpain':
                    circuit = fuse(circuit, max_qubits=1)
                if self.noise_model != None:
                    circuit = self.noise_model.apply(circuit)
                if self.exp_from_samples and self.backend.name != 'QuantumSpain':
                    circuit = self.transpile_circ(circuit)
                circuits.append(circuit)

        if self.exp_from_samples:
            if self.backend.name == 'QuantumSpain':
                results = self.backend.execute_circuit(circuits, nshots=self.nshots)
            else:
                results = [self.backend.execute_circuit(circuit, nshots=self.nshots) for circuit in circuits]
            readout_args = self.mit_kwargs['readout']
            if readout_args != {}:
                results = [apply_resp_mat_readout_mitigation(result, readout_args['response_matrix'], readout_args['ibu_iters']) for result in results]
        else:
            results = [self.backend.execute_circuit(circuit, nshots=self.nshots) for circuit in circuits]

        obs_list = []
        for result in results:
            if self.exp_from_samples:
                obs = observable.expectation_from_samples(result.frequencies())
            else:
                obs = observable.expectation(
                    result.state()
                )
            if self.obs_hardware:
                obs = np.sqrt(abs(obs))
            obs_list.append(obs)
        return obs_list


    def fits_iter(self, mit_kwargs, rand_params):
        mit_kwargs = {key: self.mit_kwargs[key] for key in ['n_training_samples', 'readout']}
        self.circuit.set_parameters(rand_params)
        circuit, observable = self.epx_value()
        data = self.mitigation['method'](
            circuit=circuit,
            observable=observable,
            noise_model=self.noise_model,
            qubit_map=self.qubit_map,
            backend=self.backend,
            nshots=self.mit_kwargs['nshots'],
            full_output=True,
            **mit_kwargs
        )
        return data

    def get_fit(self, x=None):
        rand_params = np.random.uniform(-2*np.pi,2*np.pi,int(self.nparams/2))
        mit_kwargs = {key: self.mit_kwargs[key] for key in ['n_training_samples','readout']}

        if x is None:
            data = self.fits_iter(mit_kwargs, rand_params)
            from uniplot import plot
            #plot(data[5]["noisy"]['-1'] + data[5]["noisy"]['1'],data[5]["noise-free"]['-1'] + data[5]["noise-free"]['1'])
            mean_param = data[2]
            std = data[3]
            #log.info('CDR_params '+str(mean_param)+str('err ')+str(std))

        else:
            self.inject_data(x)
            circuit, observable = self.epx_value()
            data = self.mitigation['method'](
                circuit=circuit,
                observable=observable,
                noise_model=self.noise_model,
                backend=self.backend,
                nshots=self.mit_kwargs['nshots'],
                full_output=True,
                **mit_kwargs
            )
            # TODO: is this needed? : mean_params = data[2]
            std = data[3]
        return [mean_param, std], data

    def one_mitigated_prediction(self, data, params=None):
        """This function calculates one mitigated prediction with fixed x."""
        if self.mit_params is None:
            self.mit_params = self.get_fit()[0]
        obs_list = []
        for x in data:
            obs_noisy = self.one_prediction_readout(x, params)
            obs = (1 - self.mit_params[0]) * obs_noisy / ((1 - self.mit_params[0]) ** 2 + self.mit_params[1]**2)
            obs_list.append(obs)
        return obs_list

    def step_prediction(self, data, params=None):
        if self.mitigation["step"]:
            prediction = self.one_mitigated_prediction
        else:
            prediction = self.one_prediction
        return prediction(data, params)

    def predict_sample(self):
        """This function returns all predictions."""
        if self.mitigation["final"]:
            prediction = self.one_mitigated_prediction
        else:
            prediction = self.one_prediction
        predictions = prediction(self.data)

        return predictions

    # ------------------------ PERFORMING GRADIENT DESCENT -------------------------
    # --------------------------- Parameter Shift Rule -----------------------------

    def parameter_shift(self, data):
        """This function performs the PSR for one parameter"""

        original = self.params.copy()
        params_list = []
        for i in range(len(data)):

            shifted_forward_list = []
            shifted_backward_list = []
            for parameter_index in range(self.nparams):
                shifted = self.params.copy()
                shifted[parameter_index] += (np.pi / 2) / self.scale_factors[parameter_index]
                shifted_forward_list.append(shifted)
                shifted = self.params.copy()
                shifted[parameter_index] -= (np.pi/ 2) / self.scale_factors[parameter_index]
                shifted_backward_list.append(shifted)


            params_list.append(shifted_forward_list + shifted_backward_list)

        forward_backward = self.step_prediction(data, params_list)
        forward_backward = np.reshape(forward_backward, (len(data), 2*self.nparams))


        results = np.zeros((len(data), self.nparams))

        for i in range(len(data)):
            forward = forward_backward[i][:self.nparams]
            backward = forward_backward[i][self.nparams:]
            for parameter_index in range(self.nparams):
                result = 0.5 * (forward[parameter_index] - backward[parameter_index]) * self.scale_factors[parameter_index]
                results[i, parameter_index] = result


        self.params = original

        return results

    def parameter_shift_par(self, parameter_index, x):
        """This function performs the PSR for one parameter"""

        original = self.params.copy()
        shifted = self.params.copy()

        shifted[parameter_index] += (np.pi / 2) / self.scale_factors[parameter_index]
        self.set_parameters(shifted)
        forward = self.step_prediction([x])[0]

        shifted[parameter_index] -= np.pi / self.scale_factors[parameter_index]
        self.set_parameters(shifted)
        backward = self.step_prediction([x])[0]

        self.params = original

        result = 0.5 * (forward - backward) * self.scale_factors[parameter_index]
        return result
    # ------------------------- Derivative of <O> ----------------------------------
    def circuit_derivative(self, data):
        """Derivatives of the expected value of the target observable with respect
        to the variational parameters of the circuit are performed via parameter-shift
        rule (PSR)."""

        if self.backend.name == 'numpy':
            dcirc = np.array(Parallel(n_jobs=min(self.nthreads,self.nparams))(delayed(self.parameter_shift_par)(par,x) for x in data for par in range(self.nparams)))
            dcirc = np.reshape(dcirc,(len(data), self.nparams))
        else:
            dcirc = self.parameter_shift(data)

        return dcirc

    # ---------------------- Derivative of the loss function -----------------------

    def evaluate_loss_gradients(self, data=None, labels=None):
        """This function calculates the derivative of the loss function with respect
        to the variational parameters of the model."""

        if self.noise_model is not None:
            params = self.noise_model.errors[gates.I][0][1].options
            probs = [params[k][1] for k in range(3)]
            bit_flip = self.noise_model.errors[gates.M][0][1].options[0,-1]
        else:
            probs = np.zeros(3)
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





        predictions = self.step_prediction(data)
        dcircs_data = self.circuit_derivative(data) #[self.circuit_derivative(x) for x in data]

        for i in range(len(data)):
            dcirc = dcircs_data[i,:]
            # calculate prediction
            #prediction = self.step_prediction(x)
            # derivative of E[O] with respect all thetas
            # calculate loss and dloss
            mse = predictions[i] - labels[i]
            loss += mse**2
            dloss += 2 * mse * dcirc
            if self.bp_bound:
                dloss_bound += (2 * mse * bound_grads)**2
                if labels[i] - bound_preds > 0:
                    loss_bound += (labels[i] - bound_preds)**2


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
        params = self.params - learning_rate * mhat / (np.sqrt(vhat) + epsilon)

        return m, v, loss, grads, dloss_bound, loss_bound, params

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

    def check_noise(self, circuit, observable, mit_params=None):
        if self.noise_model != None:
            circuit = self.noise_model.apply(circuit)
        if self.exp_from_samples:
            circuit = self.transpile_circ(circuit)
            result = self.backend.execute_circuit(circuit, nshots=self.nshots)
            obs =  observable.expectation_from_samples(result.frequencies())
        else:
            obs = observable.expectation(
                self.backend.execute_circuit(circuit, nshots=self.nshots).state()
            )
        if self.obs_hardware:
            obs = np.sqrt(abs(obs))
        if mit_params is not None:
            obs = (1 - mit_params[0]) * obs / ((1 - mit_params[0]) ** 2 + mit_params[1]**2)
        return obs



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
        xscale="linear"
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

        cache_dir = f"targets/{self.example}/cache"

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

        if self.evolution_model is not None:
            # to fix the evolutions
            np.random.seed(1234)
            random.seed(424242)

        # ---------------------------- Noise model utils -----------------------
        # normal sampling
        check_noise=[]
        init_params = self.params.copy()
        if self.noise_model != None:
            qm_init = self.noise_model.errors[gates.M][0][1].options[0,-1]
            noise_magnitude_init = [self.noise_model.errors[gates.I][0][1].options[j][1] for j in range(3)]

            counter = 0
            xs = [np.pi/3]*self.nqubits
            self.inject_data(xs)
            circuit, observable = self.epx_value()

            if len(self.mit_kwargs) != 0:
                circuit = error_sensitive_circuit(circuit, observable, backend = self.backend)[0]
                backend = GlobalBackend()
                exact = observable.expectation(backend.execute_circuit(circuit, nshots=self.nshots).state())
                self.mit_params, _ = self.get_fit()

        def random_step(point, var=0.005):
            """Random gaussian step on a 3D lattice."""
            new_point = []
            for dim in range(3):
                rand01 = random.random()
                if (rand01 >= 0.5):
                    sgn = +1
                else:
                    sgn = -1
                new_point.append(point[dim] + sgn * random.gauss(0, var))
            return new_point

        # ------------------------ Noise evolution -----------------------------

        if self.evolution_model is not None and self.noise_model is not None:
            # set to false if you don't want many logs
            noise_verbosity = False

            log.info(f"Noise is evolved following the model: {self.evolution_model}.")

            if self.evolution_model == "heating":
                rands = np.random.uniform(0, self.evolution_parameter, (epochs, 3))
            if self.evolution_model == "diffusion":
                rands = np.random.normal(0, self.evolution_parameter, (epochs, 3))
            if self.evolution_model == "random_walk":
                noise_magnitudes = []
                nm = noise_magnitude_init
                for _ in range(epochs):
                    nm = random_step(nm, self.evolution_parameter)
                    noise_magnitudes.append(nm)
                noise_magnitudes = np.array(noise_magnitudes)

            # support variable to compute the drift
            old_noise_magnitude = noise_magnitude_init

            # track loss bound history and noise magnitude sqrt(qx^2 + qy^2 + qz^2)
        loss_bound_evolution = []
        noise_radii = []

        # cycle over the epochs
        for epoch in range(epochs):
            if epoch%self.noise_update == 0 and epoch != 0:
                if self.noise_model is not None:
                    qm = qm_init

                    # the noise magnitude is updated according to the chosen strategy
                    if self.evolution_model == "heating" or self.evolution_model == "diffusion":
                        noise_magnitude = abs(1+rands[epoch])*np.array(old_noise_magnitude)
                    elif self.evolution_model == "random_walk":
                        noise_magnitude = noise_magnitudes[epoch]

                    if noise_verbosity:
                        log.info(f"Old params q: {old_noise_magnitude}, new: {noise_magnitude}")
                        log.info(f"Noise magnitude drift from initial: {np.sqrt(np.sum(np.array(noise_magnitude_init) - np.array(noise_magnitude))**2)}")

                    # tracking
                    loss_bound_evolution.append(bound_pred(self.layers, self.nqubits, noise_magnitude))
                    noise_radii.append(np.sqrt(np.sum(np.array(noise_magnitude_init) - np.array(noise_magnitude))**2))

                    # update the old_noise_magnitude
                    old_noise_magnitude = noise_magnitude

                    self.noise_model = generate_noise_model(qm=qm, nqubits=self.nqubits, noise_magnitude=noise_magnitude)

            if self.mitigation['step']:
                self.params = init_params
                pred = self.check_noise(circuit,observable,self.mit_params)
                check_noise.append(pred)
                if epoch != 0:
                    self.params = new_params
                    eps = abs(pred - exact)
                    #log.info(eps)
                    if eps > self.noise_threshold:
                        noisy = self.check_noise(circuit,observable)

                        mit_params_list = []
                        for _ in range(10):
                            mit_params, _ = self.get_fit()
                            mit_params_list.append(mit_params)
                        mit_params_list = np.array(mit_params_list)
                        mean_mean = np.mean(mit_params_list[:,0])
                        mean_std = np.std(mit_params_list[:,0])
                        std_mean = np.mean(mit_params_list[:,1])
                        std_std = np.std(mit_params_list[:,1])

                        partial_mean = abs(2*(1-mean_mean)**2/((1-mean_mean)**2+std_mean**2)**2 - 1/((1-mean_mean)**2+std_mean**2))
                        partial_std = abs(2*(1-mean_mean)*std_mean/((1-mean_mean)**2+std_mean**2)**2)
                        error = partial_mean*mean_std + partial_std*std_std
                        total_eps = abs(error*noisy)

                        if eps > total_eps:
                            counter += 1
                            log.info(f'## --------- Updating CDR params because eps={eps} > threshold={self.noise_threshold}!! -------- ')
                            self.mit_params[0] = mean_mean
                            self.mit_params[1] = std_mean
                            cdr_history.append(data)
                        else:
                            log.info('std='+str(total_eps)+'>'+'thr='+str(self.noise_threshold))

            iteration = 0

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
                    m, v, loss, grads, dloss_bound, loss_bound, new_params = self.apply_adam(
                        learning_rate, m, v, data, labels, iteration
                    )
                    self.params = new_params
                elif method == "Standard":
                    grads, loss = self.evaluate_loss_gradients()
                    self.params -= learning_rate * grads

                # track the training
                #print(
                log.info(
                    "Iteration "+
                    str(iteration)+
                    " epoch "+
                    str(epoch + 1)+
                    " | loss: "+
                    str(loss)
                )

                if live_plotting:
                    self.show_predictions(f"liveshow", save=True, xscale=xscale)

            grad_history.append(grads)
            loss_history.append(loss)
            if self.bp_bound:
                grad_bound_history.append(dloss_bound)
                loss_bound_history.append(loss_bound)

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


        np.save(arr=np.asarray(loss_history), file=f"{cache_dir}/loss_history_{train_type}")
        np.save(arr=np.asarray(grad_history), file=f"{cache_dir}/grad_history_{train_type}")
        if self.bp_bound:
            np.save(arr=np.asarray(grad_bound_history), file=f"{cache_dir}/grad_bound_history_{train_type}")
            np.save(arr=np.asarray(loss_bound_history), file=f"{cache_dir}/loss_bound_history_{train_type}")


        index_min = np.argmin(loss_history)

        best_params = np.load(f"{cache_dir}/params_history_{train_type}/params_epoch_{index_min + 1}.npy")

        self.params = best_params

        return loss_history, loss_bound_evolution, noise_radii

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

    def show_predictions(self, title, save=False, xscale="linear"):
        """This function shows the obtained results through a scatter plot."""

        # calculate prediction
        predictions = self.predict_sample()

        if self.ndim != 1:
            x_0_array = self.data.T[0]
        else:
            x_0_array = self.data

        if self.noise_model is not None:
            params = self.noise_model.errors[gates.I][0][1].options
            probs = [params[k][1] for k in range(3)]
            #lamb = self.noise_model.errors[gates.I][0][1].options
            #probs = (4**self.nqubits-1)*[lamb/4**self.nqubits]#[params[k][1] for k in range(3)]
            bit_flip = self.noise_model.errors[gates.M][0][1].options[0,-1]#**(1/self.nqubits)
        else:
            probs = np.zeros(3)
            bit_flip = 0

        if self.bp_bound:
            bounds = bound_pred(self.layers, self.nqubits, probs, bit_flip)

        # draw the results
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xscale(xscale)
        plt.plot(
            x_0_array,
            self.labels,
            color="orange",
            alpha=0.7,
            label="Original",
            marker="o",
            markersize=10
        )
        plt.plot(
            x_0_array,
            predictions,
            color="purple",
            alpha=0.7,
            label="Predictions",
            marker="o",
            markersize=10
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
