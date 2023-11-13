# RTQEM

Code related to `paper-link`. 
The `rtqem` algorithm implements a Real-Time Quantum Error Mitigation procedure 
to perform multi-dimensional fit in a quantum noisy scenario.

The optimization is gradient-based and makes use of the Adam optimizer.

In a few words, in the RTQEM we use a learning-based quantum error mitigation 
method (Importance Clifford Sampling) to mitigate both gradients and predictions 
during the gradient descent execution. In an evolving-noise scenario, it is possible
to set a threshold which triggers the re-learning of the noise map 
when the system has significantly changed since the last learnt map. 

A schematic representation of the algorithm follows:

![rtqem](https://github.com/qiboteam/rtqem/assets/62071516/1bdbf925-b5ea-4bf6-8790-4d27d85d72f0)


## !!! Temporary !!!
The code works now in a branch of `Qibo`, which we going to merge soon into the master branch. 

### Introduction to the usage

The code is organized as follows:
- The main functions are: `training.py` and `vqregressor.py`. The first script is called 
to train a variational quantum regressor initialized according to the second script.
- The training procedure can be set up as follows:
  - select one of the available targets, which can be found in (`src/rtqem/targets/.`);
  - once the target is defined, one can fill the target's configuration file with 
  all the desired hyper-parameters. As an example, if the `uquark` target is selected,
  the `src/rtqem/uquark.conf` file can be used to define the settings of the training. 

### Run an example of RTQEM optimization!

As an example, we provide the instructions to run the `uquark` fit with and without RTQEM.

The `uquark` configuration file is already set up to run in `RTQEM` mode in a strong-noise scenario. In order to run the optimization:

```sh
cd src/rtqem/
python training.py uquark
```

After the execution, some data will be generated:
- `./src/rtqem/liveshow.png` is a live-plotting of the fit;
- `./src/rtqem/uquark/` will contain `data.npy` and `labels.npy`, corresponding to the training input and output data.
- `src/rtqem/cache` will contain the best parameters collected during the optimization, the loss function history, the gradients history, a final plot of the predictions on the training sample and a folder in which the parameters are saved epoch by epoch.
All the described output will be saved with a label which describes the optimization configuration. More details about this can be found in `src/rtqem/savedata_utils.py`.

To run the training without RTQEM, 
edit the file `src/uquark/uquark.conf` by replacing the `mitigation` config. line 9 with:

```sh
  "mitigation": {"step":false,"final":false,"method":null, "readout":null},
```

This will result in a noisy training simulation without RTQEM.

### How to customize the training experience?

Many hyper-parameters can be used to customize the training. A detailed list follows:

- `ndim (int)`: the dimensionality of the problem. The `uquark` target, for example, is 
mono-dimensional.
- `nqubits (int)`: number of qubits used to build the parametric circuit. In our ansatz, 
this parameter must be equal to `ndim`.
- `nlayers (int)`: number of layers of the quantum machine learning model.
- `function (str)`: target name.
- `normalize_data (bool)`: if `True` the target function is normalized between [0,1].
- `noise (bool)`: if `True`, local Pauli noise is injected into the system.
- `noise_update (int)`: every `noise_update` epochs the noise changes according to
a specific evolution model.
- `noise_threshold (float)`: if this threshold is exceeded and a mitigation method is selected, the noise map is re-learned.
- `evolution_model (string)`: can be one between `"random_walk"`, `"diffusion"`, `"heating"`.
- `diffusion_parameter (float)`: mutation step of the noise into the selected noise evolution model.  
- `qm (float)`: readout noise parameter.
- `noise_magnitude ([float, float, float])`: local Pauli noise parameters (they must be positive and their sum must be lower than or equal to one).
- `bp_bound`: compute and print the bound imposed by Noise Induced Barren Plateaus according to the selected number of qubits, number of layers and noise magnitude. 
- `mitigation (dict)`: mitigation arguments:
  - `step (bool)`: if true, gradients and predictions are mitigated over training;
  - `final (bool)`: if true, the final predictions only are mitigated;
  - `method (str)`: can be `mit_obs` or `CDR`. We suggest to set `mit_obs`, since it corresponds to the last results presented in the paper. Set `null` if no mitigation is desired.
  - `readout (str)`: can be `"calibration_matrix"` (link to the paper), `"ibu"` (link to the paper) or `"randomized"` (link to the paper). Set `null` if no readout mitigation is required.
- `expectation_from_samples (bool)`: if `False`, exact simulation is performed. If `True`, shot-noise simulation is performed with set number of shots.
- `nshots (int)` number of shots for each circuit evaluation.
- `optimizer (str)` can be `"Adam"` (gradient-based) or `"cma"` (evolutionary strategy). We suggest to set `"Adam"`, since it corresponds to the last results of the paper. 
- `epochs (int)`: number of epochs of the optimization.
- `ndata (int)`: number of training data.
- `batchsize (int)`: size of the batches if mini-batch gradient descent is performed.
- `learning_rate (float)`: Adam's learning rate.
- `restart from epoch (int)`: restart from a specific training epoch using the cached parameters.
- `nthreads (int)`: number of threads on which the user wants to parallelise the code.

