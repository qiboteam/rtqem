import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qibo.config import log
from qibo import set_backend
from qibo.models.error_mitigation import calibration_matrix

from prepare_data import prepare_data
from vqregressor import vqregressor

parser = argparse.ArgumentParser()
parser.add_argument("example")

parser.add_argument(
    "--params_path",
    default=None,
    help="filepath of the best params you want to use for analysing the results",
    type=str,
)

def main(args):

    conf_file =  f"{args.example}/{args.example}.conf"
    with open(conf_file, "r") as f:
        conf = json.load(f)

    losses = []

    # --------------------------------- SETUP       

    set_backend("numpy")

    # loading data 
    data, labels1, scaler = prepare_data(
        conf["function"], 
        show_sample=False,
        normalize=conf["normalize_data"])

    # ----------------------------------- VQREGRESSOR

    VQR = vqregressor(
        layers=conf["nlayers"],
        qubit_map = conf["qubit_map"],
        data=data,
        labels=labels1,
        example=args.example,
        nqubits=conf["nqubits"],
        nshots=conf["nshots"],
        noise_model=[None, None, None],
        expectation_from_samples=conf["expectation_from_samples"],
        bp_bound=conf["bp_bound"],
        scaler=scaler,
    )

    # ------------------------------------ 
    # cycle over the different results
    
    params = np.load(f"{args.params_path}")
    VQR.set_parameters(params)
    predictions = VQR.predict_sample()

    losses = np.array(losses)
    np.save(arr=predictions, file="cleaned_predictions")

    plt.figure(figsize=(4, 4*6/8))
    plt.plot(data, predictions, label="predictions")
    plt.plot(data, labels1, label="target")
    plt.hlines(0.906, 0.0001, 1, color="black")
    plt.xscale("log")
    plt.savefig("chec_nomit.png", bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)