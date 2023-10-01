import argparse
import os
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import scienceplots

from tqdm import tqdm 
from qibo.config import log
from qibo import gates, set_backend
from qibo.models.error_mitigation import calibration_matrix, CDR
from qibo.backends import construct_backend

from prepare_data import prepare_data
from bp_utils import bound_pred, generate_noise_model
from vqregressor import vqregressor
from joblib import Parallel, delayed
import qibo

qibo.set_backend('numpy')
# plt.style.use(['science','no-latex'])

# mpl.rcParams.update({'font.size': 13})
# mpl.rcParams['xtick.major.size'] = 10
# mpl.rcParams['xtick.minor.size'] = 5
# mpl.rcParams['ytick.major.size'] = 10
# mpl.rcParams['ytick.minor.size'] = 5
# --------------------- PARSE BEST PARAMS PATH ---------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("example")

parser.add_argument(
    "--best_params_path",
    default=None,
    help="filepath of the best params you want to use for analysing the results",
    type=str,
)

parser.add_argument(
    "--conf",
    default=None,
    help="filepath of the configuration to use for analysing the results",
    type=str,
)

parser.add_argument(
    "--platform",
    default="tii1q_b1",
    help="Platform on which we perform predictions.",
    type=str,
)

parser.add_argument(
    "--run_name",
    default='',
    help="Name of the run if data are saved in a sub-folder of the example",
    type=str,
)

def loss(labels, predictions):
    """This function calculates the loss function for the entire sample."""

    # it can be useful to pass parameters as argument when we perform the cma

    loss = 0

    for prediction, label in zip(predictions, labels):
        loss += (prediction - label) ** 2

    return loss / len(labels)



def main(args):

    conf_file = (
        args.conf if args.conf is not None else f"{args.example}/{args.run_name}/{args.example}.conf"
    )
    with open(conf_file, "r") as f:
        conf = json.load(f)

    platform = conf["platform"]

    if conf["noise"]:
        qm = conf["qm"]
        noise_magnitude = conf["noise_magnitude"]
        noise = generate_noise_model(qm=qm, nqubits=conf["nqubits"], noise_magnitude=noise_magnitude)
    else:
        noise = None

    noise_setting = noise

    data, labels, scaler = prepare_data(
    conf["function"], 
    show_sample=False,
    normalize=conf["normalize_data"], 
    run_name=args.run_name)
    

    if len(np.shape(data)) != 1:
        data1 = data.T[0]
    else:
        data1 = data


    if conf["qibolab"]:        
        backend = construct_backend("qibolab", conf["platform"])
        backend.transpiler = None
    else:
        set_backend('numpy')
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
        "CDR": {"n_training_samples": 10, "readout": readout, "N_update": 20, "N_mean": 10, "nshots":10000},
        "vnCDR": {
            "n_training_samples": 10,
            "noise_levels": np.arange(3),
            "insertion_gate": "RX",
            "readout": readout,
        },
        None: {},
    }   

    mitigation = {"step":True,"final":True,"method":"CDR","readout":None}

    VQR = vqregressor(
        layers=conf["nlayers"],
        data=data,
        labels=labels,
        example=args.example,
        nqubits=conf["nqubits"],
        backend=backend,
        nshots=conf["nshots"],
        expectation_from_samples=conf["expectation_from_samples"],
        noise_model=[noise_setting,None],
        bp_bound=conf["bp_bound"],
        mitigation=mitigation,
        mit_kwargs=mit_kwargs[mitigation["method"]],
        scaler=scaler,
    )
    VQR.mit_params = VQR.get_fit()[0]

    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(5*2/3, 5*(6/8)*2/3))

    run_names = ['All_3', '10_3', '20_3', '30_3', '40_3', '50_3', '60_3', '70_3', '80_3', '90_3', '100_3', '_no_3']

    loss_list=[]
    for run_name in run_names:

        means = np.load(f"{args.example}/benchmark_4_new_update{run_name}/means_{platform}_realtime_mitigation_step_yes_final_yes.npy")

        loss_list.append(loss(labels,means))
    
    fit_axis.plot(list(range(len(run_names))),loss_list)
    fit_fig.savefig(f"{args.example}/noise_ev.pdf", bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)


    