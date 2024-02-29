# some useful python package
import argparse
import json
import os
import time
from pathlib import Path

# extra dependencies
import numpy as np

# qibo's
from qibo import gates, set_backend
from qibo.backends import construct_backend
from qibo.models.error_mitigation import get_response_matrix

# rtqem 
from bp_utils import bound_pred, generate_noise_model
from prepare_data import prepare_data
from savedata_utils import get_training_type
from vqregressor import VQRegressor

parser = argparse.ArgumentParser(description="Training the vqregressor")
parser.add_argument("example")

args = parser.parse_args()


if args.example[-1] == "/":
    args.example = args.example[:-1]

cache_dir = f"targets/{args.example}/cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

with open("targets/{}/{}.conf".format(args.example, args.example), "r") as f:
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
        bit_flip = noise.errors[gates.M][0][1].options[0,-1]
        bounds = bound_pred(layers, nqubits, probs, bit_flip)
        print('bound', bounds)
        np.save(f"{cache_dir}/pred_bound", np.array(bounds))
else:
    print("Noisless model is executed.")
    noise = None

if conf["qibolab"]:
    runcard = conf["runcard"]
    if runcard:
        runcard = Path(runcard)
    backend = construct_backend("qibolab", conf["platform"], runcard=runcard)
    backend.transpiler = None
else:
    set_backend('numpy')
    #set_threads(5)
    backend = construct_backend('numpy')
    #backend.set_threads(5)
    
readout = {}
if conf["mitigation"]["readout"] is not None:
    if conf["mitigation"]["readout"][0] == "response_matrix":
        if conf["mitigation"]["readout"][1] == "ibu_iters":
            ibu_iters = 20
        else:
            ibu_iters = None
        resp_m = get_response_matrix(
            1, qubit_map=conf["qubit_map"], backend=backend, noise_model=noise, nshots=10000
        )
        np.save(f"{cache_dir}/resp_matrix.npy", resp_m)
        readout["response_matrix"] = resp_m
        readout["ibu_iters"] = ibu_iters
    elif conf["mitigation"]["readout"] == "randomized":
        readout["ncircuits"] = 10
    else:
        raise AssertionError("Invalid readout mitigation method specified.")

mit_kwargs = {
    "CDR": {"n_training_samples": 5, "readout": readout, "N_update": 0, "nshots": 10000},
    "ICS": {"n_training_samples": 20, "readout": readout, "nshots": 10000},
    None: {},
}

VQR = VQRegressor(
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
    history, _, _ = VQR.gradient_descent(
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


print(f"Execution time required: ", (end - start))


VQR.show_predictions(f"targets/{args.example}/predictions_{conf['optimizer']}", save=True, xscale=conf["xscale"])
np.save(f"{cache_dir}/best_params_{conf['optimizer']}_{training_type}", VQR.params)
