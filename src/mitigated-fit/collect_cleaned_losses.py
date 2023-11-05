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
    "--collection_path",
    default=None,
    help="filepath of the best params you want to use for analysing the results",
    type=str,
)

updates = ["0075", "005", "0025", "00"]

def main(args):

    conf_file =  f"{args.collection_path}/{args.example}.conf"
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
        noise_model=[None, None, None, None, None],
        expectation_from_samples=conf["expectation_from_samples"],
        bp_bound=conf["bp_bound"],
        scaler=scaler,
    )

    # ------------------------------------ 
    # cycle over the different results
    for run in updates:
        this_loss = []
        print(f"update every {run}")
        for i in tqdm(range(100)):
            params = np.load(f"{args.collection_path}/evol_{run}/cache/params_history_realtime_mitigation_step_yes_final_yes/params_epoch_{i+1}.npy")
            this_loss.append(VQR.loss(params=params))
        losses.append(this_loss)
    
    losses = np.array(losses)
    np.save(arr=losses, file="cleaned_losses")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)