import argparse
import os
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scienceplots

from tqdm import tqdm 

from qibo.models.error_mitigation import calibration_matrix
from qibo.backends import construct_backend

from prepare_data import prepare_data
from bp_utils import generate_noise_model
from vqregressor import vqregressor


plt.style.use('science')

mpl.rcParams.update({'font.size': 22})
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.minor.size'] = 5
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


def plot(fit_axis, loss_grad_axes, data, means, stds, loss_history, loss_bound_history, grad_history, grad_bound_history, color, label):

    global ndata, nruns
    
    if len(np.shape(data)) != 1:
        data = data.T[0]

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
    loss_grad_axes[0].set_yscale('log')
    loss_grad_axes[1].set_yscale('log')
    loss_grad_axes[1].plot(
        np.max(np.sqrt((grad_history*grad_history)),axis=-1), 
        c=color,
        lw=2,
        alpha=0.7,
        label=label)
    if type(loss_history) == np.ndarray and type(loss_bound_history) == np.ndarray:
        if label == "No mitigation":
            loss_grad_axes[0].plot(loss_bound_history, '--', c='black', lw=2, alpha=0.7, label='BP bound')
        elif label == "Exact":
            loss_grad_axes[1].plot(
                grad_bound_history, 
                '--',
                c='black',
                lw=2,
                alpha=0.7,
                label='BP bound')
    
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

    # noise parameters
    qm = conf["qm"]
    noise_magnitude = conf["noise_magnitude"]

    # noise model
    if conf["noise"]:
        noise = generate_noise_model(qm=qm, nqubits=conf["nqubits"], noise_magnitude=noise_magnitude)
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

    if len(np.shape(data)) != 1:
        data1 = data.T[0]
    else:
        data1 = data

    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(10, 6))
    fit_axis.plot(data1, labels, c="black", lw=2, alpha=0.8, label="Target function")
    fit_axis.set_title("Statistics on results")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")
    #fit_fig.legend(loc=1, borderaxespad=3)
    if conf["bp_bound"]:
        pred_bound = np.load(f"{args.example}/cache/pred_bound.npy")
        fit_axis.plot(data1, [pred_bound]*len(data), '--', c="black", alpha=0.7, lw=2, label="BP bound")

    loss_grad_fig , loss_grad_axes = plt.subplots(2, 1, figsize=(10,12))
    plt.rcParams['text.usetex'] = True
    loss_grad_axes[0].set_title('Loss history')
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
        if f"best_params_{conf['optimizer']}_realtime_mitigation_step_yes_final_yes" in f:
            settings.append("realtime_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":True,"final":True,"method":"CDR","readout":None})
            colors.append('red')
            labels.append('Real time mitigation')
        if f"best_params_{conf['optimizer']}_noiseless" in f:
            settings.append("noiseless")
            mitigation_settings.append({"step":False,"final":False,"method":None,"readout":None})
            colors.append('green')
            labels.append('Exact')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_yes_final_yes" in f:
            settings.append("full_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":False,"final":True,"method":"CDR","readout":"calibration_matrix"})
            colors.append('orange')
            labels.append('Full mitigation')

    for setting, mitigation, color, label in zip(settings, mitigation_settings, colors, labels):

        print(f"> Drawing '{setting}' plot in {color}.")
        print(f"> Loading best parameters from:\n  -> '{args.example}/cache/best_params_{conf['optimizer']}_{setting}.npy'.")
        best_params = np.load(f"{args.example}/cache/best_params_{conf['optimizer']}_{setting}.npy")
        if setting == 'noiseless':
            noise_setting = None
        else:
            noise_setting = noise
        # initialize vqr with data and best parameters
        VQR = vqregressor(
            layers=conf["nlayers"],
            data=data,
            labels=labels,
            example=args.example,
            nqubits=conf["nqubits"],
            backend=backend,
            nshots=conf["nshots"],
            expectation_from_samples=conf["expectation_from_samples"],
            noise_model=noise_setting,
            bp_bound=conf["bp_bound"],
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
        
        
        # TO DO: ADD FUNCTION WHICH CLASSIFIES THE TRAINING
        np.save(arr=means, file=f"{args.example}/means_{platform}")
        np.save(arr=stds, file=f"{args.example}/stds_{platform}")

        loss_history = np.load(f"{args.example}/cache/loss_history_{setting}.npy")
        print('Minimum loss', np.argmin(loss_history) + 1)
        grad_history = np.load(f"{args.example}/cache/grad_history_{setting}.npy")

        if conf["bp_bound"]:
            loss_bound_history = np.load(f"{args.example}/cache/loss_bound_history_{setting}.npy")
            grad_bound_history = np.load(f"{args.example}/cache/grad_bound_history_{setting}.npy")
        else:
            loss_bound_history = 0
            grad_bound_history = 0

        plot(
            fit_axis,
            loss_grad_axes,
            data,
            means,
            stds,
            loss_history,
            loss_bound_history,
            grad_history,
            grad_bound_history,
            color,
            label
        )

    fit_axis.minorticks_off()
    loss_grad_axes[0].minorticks_off()
    loss_grad_axes[1].minorticks_off()
    fit_axis.legend(loc=1)
    fit_fig.savefig('fits_benchmark.pdf', bbox_inches='tight')
    loss_grad_axes[1].legend(loc=1)
    loss_grad_fig.savefig('gradients_analysis.pdf', bbox_inches='tight')
    


# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)
