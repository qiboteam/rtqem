# some useful python package
import argparse
import json
import os
import time
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt 

import numpy as np
from bp_utils import bound_pred, generate_noise_model
from prepare_data import prepare_data
from qibo import gates, set_backend, set_threads
from qibo.backends import construct_backend
from qibo.models.error_mitigation import calibration_matrix
from qibo.noise import NoiseModel, PauliError, ReadoutError
from savedata_utils import get_training_type
#from uniplot import plot
from vqregressor import vqregressor

parser = argparse.ArgumentParser(description="Training the vqregressor")
parser.add_argument("example")

args = parser.parse_args()


if args.example[-1] == "/":
    args.example = args.example[:-1]

cache_dir = f"{args.example}/cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

with open("{}/{}.conf".format(args.example, args.example), "r") as f:
    conf = json.loads(f.read())

# model definition
nqubits = conf["nqubits"]
layers = conf["nlayers"]
ndata = conf["ndata"]

# get string to identify the training type
training_type = get_training_type(conf["mitigation"], conf["noise"])

# prepare data
data, labels, scaler = prepare_data(conf["function"], normalize=conf["normalize_data"], show_sample=False)

# noise parameters
qm = conf["qm"]
noise_magnitude = conf["noise_magnitude"]

# noise model
if conf["noise"]:
    print("Generating noise model given noise paramaters.")
    noise = generate_noise_model(qm=qm, nqubits=nqubits, noise_magnitude=noise_magnitude)
    if conf["bp_bound"]:
        params = noise.errors[gates.I][0][1].options
        probs = [params[k][1] for k in range(3)]
        #lamb = noise.errors[gates.I][0][1].options
        #probs = (4**nqubits-1)*[lamb/4**nqubits]
        bit_flip = noise.errors[gates.M][0][1].options[0,-1]
        bounds = bound_pred(layers, nqubits, probs, bit_flip)
        print('bound', bounds)
        np.save(f"{cache_dir}/pred_bound", np.array(bounds))
else:
    print("Noisless model is executed.")
    noise = None

if conf["qibolab"]:    
    backend = construct_backend("qibolab", conf["platform"])
    backend.transpiler = None
else:
    set_backend('numpy')
    #set_threads(5)
    backend = construct_backend('numpy')
    #backend.set_threads(5)
    
readout = {}
if conf["mitigation"]["readout"] is not None:
    if conf["mitigation"]["readout"][0] == "calibration_matrix":
        if conf["mitigation"]["readout"][1] == "ibu":
            inv = False
        else:
            inv = True
        cal_m = calibration_matrix(
            1, qubit_map=conf["qubit_map"], inv=inv, backend=backend, noise_model=noise, nshots=10000
        )
        np.save(f"{cache_dir}/cal_matrix.npy", cal_m)
        readout["calibration_matrix"] = cal_m
        readout["inv"] = inv
    elif conf["mitigation"]["readout"] == "randomized":
        readout["ncircuits"] = 10
    else:
        raise AssertionError("Invalid readout mitigation method specified.")

mit_kwargs = {
    "CDR": {"n_training_samples": 5, "readout": readout, "N_update": 0, "nshots": 10000},
    "mit_obs": {"n_training_samples": 10, "readout": readout, "nshots": 10000},
    None: {},
}

VQR = vqregressor(
    nqubits=nqubits,
    qubit_map = conf["qubit_map"],
    layers=layers,
    data=data,
    labels=labels,
    nshots=conf["nshots"],
    expectation_from_samples=conf["expectation_from_samples"],
    obs_hardware=conf["obs_hardware"],
    backend=backend,
    nthreads=conf["nthreads"],
    noise_model=[noise, conf["noise_update"], conf["noise_threshold"], conf["evolution_model"], conf["diffusion_parameter"]],
    bp_bound=conf["bp_bound"],
    mitigation=conf["mitigation"],
    mit_kwargs=mit_kwargs[conf["mitigation"]["method"]],
    scaler=scaler,
    example=args.example,
)

start = time.time()
if conf["optimizer"] == "Adam":
    # set the training hyper-parameters
    epochs = conf["epochs"]
    learning_rate = conf["learning_rate"]
    # perform the training
    history, bound_history, noise_radii = VQR.gradient_descent(
        learning_rate=learning_rate,
        epochs=epochs,
        restart_from_epoch=conf["restart_from_epoch"],
        batchsize=conf["batchsize"],
        method="Adam",
        J_treshold=2**conf["nqubits"]/conf["nshots"],
        xscale=conf["xscale"]
    )
elif conf["optimizer"] == "CMA":
    VQR.cma_optimization()
end = time.time()

predictions = VQR.predict_sample()

#plot([labels, predictions], legend_labels=["target", "predictions"])

print(f"Execution time required: ", (end - start))

# best_params = np.load('gluon/best_params_Adam.npy',allow_pickle=True)
# VQR.params = best_params

VQR.show_predictions(f"{args.example}/predictions_{conf['optimizer']}", save=True)
np.save(f"{cache_dir}/best_params_{conf['optimizer']}_{training_type}", VQR.params)

np.save(arr=bound_history, file="bounds")
np.save(arr=noise_radii, file="noise_radii")