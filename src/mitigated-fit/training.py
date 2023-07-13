# some useful python package
import argparse
import json
import os
import time
from functools import reduce
from itertools import product

import numpy as np
from bp_utils import bound_pred, generate_noise_model
from prepare_data import prepare_data
from qibo import gates
from qibo.backends import construct_backend
from qibo.models.error_mitigation import calibration_matrix
from qibo.noise import DepolarizingError, NoiseModel, PauliError, ReadoutError
from savedata_utils import get_training_type
from uniplot import plot
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
data, labels, scaler = prepare_data(conf["function"], show_sample=False)

# noise parameters
qm = conf["qm"]
noise_magnitude = conf["noise_magnitude"]

# noise model
if conf["noise"]:
    print("Generating noise model given noise paramaters.")
    noise = generate_noise_model(qm=qm, nqubits=nqubits, noise_magnitude=noise_magnitude)
else:
    print("Noisless model is executed.")
    noise = None

if conf["qibolab"]:
    backend = construct_backend("qibolab", conf["platform"])
else:
    backend = construct_backend("numpy")
    backend.set_threads(1)
    
readout = {}
if conf["mitigation"]["readout"] is not None:
    if conf["mitigation"]["readout"] == "calibration_matrix":
        cal_m = calibration_matrix(
            1, backend=backend, noise_model=noise, nshots=conf["nshots"]
        )
        np.save(f"{cache_dir}/cal_matrix.npy", cal_m)
        readout["calibration_matrix"] = cal_m
    elif conf["mitigation"]["readout"] == "randomized":
        readout["ncircuits"] = 10
    else:
        raise AssertionError("Invalid readout mitigation method specified.")

mit_kwargs = {
    "ZNE": {"noise_levels": np.arange(5), "insertion_gate": "RX", "readout": readout},
    "CDR": {"n_training_samples": 10, "readout": readout, "N_update": 20, "N_mean": 10},
    "vnCDR": {
        "n_training_samples": 10,
        "noise_levels": np.arange(3),
        "insertion_gate": "RX",
        "readout": readout,
    },
    None: {},
}

VQR = vqregressor(
    nqubits=nqubits,
    layers=layers,
    data=data,
    labels=labels,
    nshots=conf["nshots"],
    expectation_from_samples=conf["expectation_from_samples"],
    obs_hardware=conf["obs_hardware"],
    backend=backend,
    noise_model=noise,
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
    history = VQR.gradient_descent(
        learning_rate=learning_rate,
        epochs=epochs,
        restart_from_epoch=conf["restart_from_epoch"],
        batchsize=conf["batchsize"],
        method="Adam",
        J_treshold=2/conf["nshots"],
    )
elif conf["optimizer"] == "CMA":
    VQR.cma_optimization()
end = time.time()

predictions = VQR.predict_sample()

plot([labels, predictions], legend_labels=["target", "predictions"])

print(f"Execution time required: ", (end - start))

# best_params = np.load('gluon/best_params_Adam.npy',allow_pickle=True)
# VQR.params = best_params

VQR.show_predictions(f"{args.example}/predictions_{conf['optimizer']}", save=True)
np.save(f"{cache_dir}/best_params_{conf['optimizer']}_{training_type}", VQR.params)

if conf["noise"] and conf["bp_bound"] and os.path.exists(f"{cache_dir}/pred_bound") == False:
    params = noise.errors[gates.I][0][1].options
    probs = [params[k][1] for k in range(4**nqubits-1)]
    bit_flip = noise.errors[gates.M][0][1].options[0,-1]**(1/nqubits)
    bounds = bound_pred(layers, nqubits, probs, bit_flip)
    np.save(f"{cache_dir}/pred_bound", np.array(bounds))