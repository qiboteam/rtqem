# some useful python package
import argparse
import json
import os
import random
import time

import numpy as np
import scipy.stats
from qibo import gates
from qibo.backends import construct_backend
from qibo.models.error_mitigation import calibration_matrix
from qibo.noise import DepolarizingError, NoiseModel
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
training_type = get_training_type(conf)

# random data
data = np.linspace(-1, 1, ndata)
scaler = lambda x: x
# labeling them
if conf["function"] == "sinus":
    labels = np.sin(2 * data)
elif conf["function"] == "hdw_target":
    labels = np.exp(-data) * np.cos(3 * data) * 0.3
elif conf["function"] == "gamma":
    labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
elif conf["function"] == "gluon":
    scaler = lambda x: np.log(x)
    parton = conf["parton"]
    data = np.loadtxt(f"gluon/data/{parton}.dat")
    idx = random.sample(range(len(data)), ndata)
    labels = data.T[1][idx]
    data = data.T[0][idx]

# noise model
if conf["noise"]:
    noise = NoiseModel()
    noise.add(DepolarizingError(lam=0.1), gates.RX)
else:
    noise = None

if conf["qibolab"]:
    backend = construct_backend("qibolab", conf["platform"])
else:
    backend = construct_backend("numpy")

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
    "CDR": {"n_training_samples": 10, "readout": readout, "N_update": 10, "N_mean": 4},
    "vnCDR": {
        "n_training_samples": 10,
        "noise_levels": np.arange(3),
        "insertion_gate": "RX",
        "readout": readout,
    },
    None: {},
}

VQR = vqregressor(
    layers=layers,
    data=data,
    labels=labels,
    nshots=conf["nshots"],
    expectation_from_samples=conf["expectation_from_samples"],
    obs_hardware=conf["obs_hardware"],
    backend=backend,
    noise_model=noise,
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
