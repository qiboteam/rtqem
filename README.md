# RTQEM

Code related to `paper-link`. 
The `rtqem` algorithm implements a Real-Time Quantum Error Mitigation procedure 
to perform multi-dimensional fit in a quantum noisy scenario.

The optimization is gradient-based and makes use of Adam optimizer.

In a few words, in the RTQEM we use a learning-based quantum error mitigation 
method (Importance Clifford Sampling) to mitigate both gradients and predictions 
during the gradient descent execution. In an evolving-noise scenario, it is possible
to set a threshold which govern the possibility of re-learning the noise map 
if the system is changed too much from the last time the map was learn. 

A schematic representation of the algorithm follows:

[rtqem.pdf](https://github.com/qiboteam/mitigated-fit/files/13299448/rtqem.pdf)


### Introduction to the usage

The code is organized as follows:
- The main functions are: `training.py` and `vqregressor.py`. The first script is called 
to train a variational quantum regressor initialized according to the second script.
- The training procedure can be set up as follows:
  - select one of the available targets, which can be found in (`src/rtqem/targets/.`);
  - once the target is defined, one can fill the target's configuration file with 
  all the desired hyper-parameters. As an example, if the `uquark` target is selected,
  the `src/rtqem/uquark.conf` file can be used to define the settings of the training. 

### Hyper-parameters into configuration files

Many hyper-parameters can be used to customize the training. A detailed list follows:

- `ndim (int)`: the dimensionality of the problem. The `uquark` target, for example, in 
mono-dimensional.
- `nqubits (int)`: number of qubits used to build the parametric circuit. In our ansatz, 
this parameter must be equal to `ndim`.
- `nlayers (int)`: number of layers of the quantum machine learning model.
- `function (str)`: target name.
- `normalize_data (bool)`: target function output is normalized between [0,1].
- `noise (bool)`: if `True`, local Pauli noise is injected into the system.
- `noise_update (int)`: every `noise_update` epochs the noise changes according to
a specific evolution model.
- `noise_threshold (float)`: if this threshold is exceeded and a mitigation method is selected, the noise map is re-learned.
- `obs_hardware (bool)`: computing expectation values directly on frequencies (use only if working on hardware).
- `evolution_model (string)`: can be one between `"random_walk"`, `"diffusion"`, `"heating"`.
- `diffusion_parameter (float)`: mutation step of the noise into the selected noise evolution model.  



