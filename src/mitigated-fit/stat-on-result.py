import argparse
import random
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import scienceplots

import scipy.stats
from tqdm import tqdm 

from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates
from qibo.models.error_mitigation import calibration_matrix
from qibo.backends import construct_backend

from prepare_data import prepare_data
from vqregressor import vqregressor

plt.style.use('science')

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


# ---------------------- MAIN FUNCTION -----------------------------------------

ndata = 50
nruns = 10

def plot(fit_axis, loss_grad_axes, data, means, stds, loss_history, grad_history, color, label):

    global ndata, nruns 
    
    # plot results
    fit_axis.plot(data, means, c=color, alpha=0.7, lw=2, label=label)
    fit_axis.fill_between(
        data,
        means - stds,
        means + stds,
        alpha=0.25,
        color=color,
    )
    #fit_fig.legend(loc=1, borderaxespad=3)
    #fit_fig.savefig('fits_benchmark.pdf', bbox_inches='tight')

    loss_grad_axes[0].plot(loss_history, c=color, lw=2, alpha=0.7, label=label)
    loss_grad_axes[1].plot(
        np.sqrt((grad_history*grad_history).sum(-1)), 
        c=color,
        lw=2,
        alpha=0.7,
        label=label)
    
    #loss_grad_fig.legend(loc=1, borderaxespad=3)
    #loss_grad_fig.savefig('gradients_analysis.pdf', bbox_inches='tight')

    
    
def main(args):
    
    conf_file = (
        args.conf if args.conf is not None else f"{args.example}/{args.example}.conf"
    )
    with open(conf_file, "r") as f:
        conf = json.load(f)

    platform = conf["platform"]
    if conf["qibolab"]:
        backend = construct_backend("qibolab", conf["platform"])
    else:
        #backend = construct_backend("qibojit", platform="numba")
        backend = construct_backend("numpy")

    # define dataset cardinality and number of executions
    global ndata, nruns

    # loading data 
    data, labels, scaler = prepare_data(conf["function"], show_sample=False)

    # noise model
    if conf["noise"]:
        noise = NoiseModel()
        noise.add(DepolarizingError(lam=0.1), gates.RX)
    else:
        noise = None

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
        "CDR": {"n_training_samples": 10, "readout": readout, "N_update": 1, "N_mean": 1},
        "vnCDR": {
            "n_training_samples": 10,
            "noise_levels": np.arange(3),
            "insertion_gate": "RX",
            "readout": readout,
        },
        None: {},
    }   

    # plot results
    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(10, 6))
    fit_axis.plot(data, labels, c="black", lw=2, alpha=0.8, label="Target function")
    fit_axis.set_title("Statistics on results")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")
    #fit_fig.legend(loc=1, borderaxespad=3)

    loss_grad_fig , loss_grad_axes = plt.subplots(2, 1, figsize=(10,8))
    plt.rcParams['text.usetex'] = True
    loss_grad_axes[0].set_title('Loss history')
    loss_grad_axes[0].set_xlabel('Epoch')
    loss_grad_axes[0].set_ylabel("Loss")
    loss_grad_axes[1].set_title(r'$\|Grad\|$ history')
    loss_grad_axes[1].set_xlabel('Epoch')
    loss_grad_axes[1].set_ylabel(r'$\|Grad\|$')
    #loss_grad_fig.legend(loc=1, borderaxespad=3)
    
    files = os.listdir(f"{args.example}/cache/")
    settings, mitigation_settings, colors, labels = [], [], [], []
    for f in files:
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            mitigation_settings.append({"step":False,"final":False,"method":None,"readout":None})
            colors.append('blue')
            labels.append('No mitigation')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_yes_final_yes" in f:
            settings.append("full_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":True,"final":True,"method":"CDR","readout":"calibration_matrix"})
            colors.append('red')
            labels.append('Full mitigation')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_no_final_yes" in f:
            settings.append("full_mitigation_step_no_final_yes")
            mitigation_settings.append({"step":False,"final":True,"method":"CDR","readout":"calibration_matrix"})
            colors.append('orange')
            labels.append('Mitigation on predictions')


    for setting, mitigation, color, label in zip(settings, mitigation_settings, colors, labels):

        print(f"> Drawing '{setting}' plot in {color}.")
        print(f"> Loading best parameters from:\n  -> '{args.example}/cache/best_params_{conf['optimizer']}_{setting}.npy'.")
        best_params = np.load(f"{args.example}/cache/best_params_{conf['optimizer']}_{setting}.npy")

        # initialize vqr with data and best parameters
        VQR = vqregressor(
            layers=conf["nlayers"],
            data=data,
            labels=labels,
            example=args.example,
            nshots=conf["nshots"],
            expectation_from_samples=conf["expectation_from_samples"],
            noise_model=noise,
            mitigation=mitigation,
            mit_kwargs=mit_kwargs[mitigation["method"]],
            scaler=scaler,
        )
        VQR.set_parameters(best_params)

        predictions = []

        for _ in tqdm(range(nruns)):
            predictions.append(VQR.predict_sample())

        predictions = np.asarray(predictions)

        means = predictions.mean(0)
        stds = predictions.std(0)

        print(stds)
        
        # TO DO: ADD FUNCTION WHICH CLASSIFIES THE TRAINING
        np.save(arr=means, file=f"{args.example}/means_{platform}")
        np.save(arr=stds, file=f"{args.example}/stds_{platform}")

        loss_history = np.load(f"{args.example}/cache/loss_history_{setting}.npy")
        grad_history = np.load(f"{args.example}/cache/grad_history_{setting}.npy")

        plot(
            fit_axis,
            loss_grad_axes,
            data,
            means,
            stds,
            loss_history,
            grad_history,
            color,
            label
        )

    fit_axis.legend(loc=1)
    fit_fig.savefig('fits_benchmark.pdf', bbox_inches='tight')
    loss_grad_axes[0].legend(loc=1)
    loss_grad_fig.savefig('gradients_analysis.pdf', bbox_inches='tight')
    


# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)
