import argparse
import os
import json
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt

# qibo's
import qibo
from qibo.config import log
from qibo import gates, set_backend
from qibo.models.error_mitigation import calibration_matrix
from qibo.backends import construct_backend

# rtqem's
from prepare_data import prepare_data
from bp_utils import bound_pred, generate_noise_model
from vqregressor import VQRegressor

qibo.set_backend('numpy')

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

# ---------------------- MAIN FUNCTION -----------------------------------------

ndata = 30
nruns = 20


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
        alpha=0.2,
        hatch="//",
        color=color,
    )

    if label != "Mitigation after training":
        loss_grad_axes[0].plot(loss_history, c=color, lw=2, alpha=0.7, label=label)
        loss_grad_axes[0].set_yscale('log')
        loss_grad_axes[1].set_yscale('log')
        loss_grad_axes[1].plot(
            np.mean(np.sqrt((grad_history*grad_history)),axis=-1), 
            c=color,
            lw=2,
            alpha=0.7,
            label=label)

    
    
def main(args):
    
    conf_file = (
        args.conf if args.conf is not None else f"targets/{args.example}/{args.run_name}/{args.example}.conf"
    )
    with open(conf_file, "r") as f:
        conf = json.load(f)

    platform = conf["platform"]
        
    # define dataset cardinality and number of executions
    global ndata, nruns

    # loading data 
    data, labels1, scaler = prepare_data(
        conf["function"], 
        show_sample=False,
        normalize=conf["normalize_data"], 
        run_name=args.run_name)

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
        #backend.transpiler = None
    else:
        set_backend('numpy')
        backend = construct_backend("numpy")

    readout = {}
    if conf["mitigation"]["readout"] is not None:
        if conf["mitigation"]["readout"][0] == "calibration_matrix":
            if conf["mitigation"]["readout"][1] == "ibu":
                inv = False
            else:
                inv = True
            cal_m = calibration_matrix(
                1, qubit_map=conf["qubit_map"], inv=inv, backend=backend, noise_model=noise, nshots=100000
            )
            log.info(cal_m)
            np.save(f"{cache_dir}/cal_matrix.npy", cal_m)
            readout["calibration_matrix"] = cal_m
            readout["inv"] = inv
        elif conf["mitigation"]["readout"] == "randomized":
            readout["ncircuits"] = 10
        else:
            raise AssertionError("Invalid readout mitigation method specified.")

    mit_kwargs = {
        "CDR": {"n_training_samples": 5, "readout": readout, "N_update": 0, "nshots": 1000},
        "mit_obs": {"n_training_samples": 10, "readout": readout, "nshots": 10000},
        None: {},
    }

    # plot results

    if len(np.shape(data)) != 1:
        data1 = data.T[0]
    else:
        data1 = data

    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(5*2/3, 5*(6/8)*2/3))
    fit_axis.plot(data1, labels1, c="black", lw=2, alpha=0.8, label="Target function")
    fit_axis.set_title("Statistics on results")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")
    #fit_fig.legend(loc=1, borderaxespad=3)
    loss_bound_history = 0
    if conf["bp_bound"]:
        params = noise.errors[gates.I][0][1].options
        probs = [params[k][1] for k in range(3)]
        bit_flip = noise.errors[gates.M][0][1].options[0,-1]#**(1/conf['nqubits'])
        pred_bound = bound_pred(conf['nlayers'], conf['nqubits'], probs, bit_flip)
        fit_axis.plot(data1, [pred_bound]*len(data), '--', c="black", alpha=0.7, lw=2, label="BP bound")

        loss_bound_history = 0
        for y in labels:
            if y - pred_bound > 0:
                loss_bound_history += (y - pred_bound)**2
        loss_bound_history /= len(labels)
        log.info(str(loss_bound_history))

    loss_grad_fig , loss_grad_axes = plt.subplots(2, 1, figsize=(5*2/3, 5*(8/6)*2/3))
    loss_grad_axes[0].set_title('Loss history')
    loss_grad_axes[0].set_ylabel("Loss")
    loss_grad_axes[1].set_title('Grad history')
    loss_grad_axes[1].set_xlabel('Epoch')
    loss_grad_axes[1].set_ylabel('Grad')
    #loss_grad_fig.legend(loc=1, borderaxespad=3)
    
    files = os.listdir(f"targets/{args.example}/{args.run_name}/cache/")
    settings, mitigation_settings, colors, labels = [], [], [], []
    for f in files:
        if f"best_params_{conf['optimizer']}_noiseless" in f:
            settings.append("noiseless")
            mitigation_settings.append({"step":False,"final":False,"method":None,"readout":None})
            colors.append('green')
            labels.append('Noiseless')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            mitigation_settings.append({"step":False,"final":False,"method":None,"readout":None})
            colors.append('blue')
            labels.append('No mitigation')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            mitigation_settings.append({"step":False,"final":True,"method":"mit_obs","readout":None})
            colors.append('orange')
            labels.append('Mitigation after training')
        if f"best_params_{conf['optimizer']}_realtime_mitigation_step_yes_final_yes" in f:
            settings.append("realtime_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":True,"final":True,"method":"mit_obs","readout":None})
            colors.append('red')
            labels.append('Real time mitigation')
        if f"best_params_{conf['optimizer']}_realtime_mitigation_step_yes_final_yes" in f:
            settings.append("realtime_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":True,"final":False,"method":"mit_obs","readout":None})
            colors.append('red')
            labels.append('Mitigation training')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_yes_final_yes" in f:
            settings.append("full_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":False,"final":True,"method":"mit_obs","readout":"calibration_matrix"})
            colors.append('orange')
            labels.append('Full mitigation')

    for setting, mitigation, color, label in zip(settings, mitigation_settings, colors, labels):

        print(f"> Drawing '{setting}' plot in {color}.")
        print(f"> Loading best parameters from:\n  -> 'targets/{args.example}/cache/best_params_{conf['optimizer']}_{setting}.npy'.")


        if setting == 'noiseless':
            noise_setting = None
        else:
            noise_setting = noise


        VQR = VQRegressor(
            layers=conf["nlayers"],
            qubit_map = conf["qubit_map"],
            data=data,
            labels=labels1,
            example=args.example,
            nqubits=conf["nqubits"],
            backend=backend,
            nshots=conf["nshots"],
            expectation_from_samples=conf["expectation_from_samples"],
            noise_model=[noise_setting,None,None],
            bp_bound=conf["bp_bound"],
            mitigation=mitigation,
            mit_kwargs=mit_kwargs[mitigation["method"]],
            scaler=scaler,
        )
        


        #Noise fixed
        loss_history = np.load(f"targets/{args.example}/{args.run_name}/cache/loss_history_{setting}.npy")
        best_params = np.load(f"targets/{args.example}/{args.run_name}/cache/best_params_{conf['optimizer']}_{setting}.npy")       


        VQR.set_parameters(best_params)

        predictions = []

        VQR.mit_params = VQR.get_fit()[0]

        def get_pred(j):
            pred = VQR.predict_sample()
            return pred

        if backend.name == 'numpy':
            pred = Parallel(n_jobs=min(conf["nthreads"],nruns))(delayed(get_pred)(j) for j in range(nruns))
        else:
            pred = []
            for j in range(nruns):
                log.info('run '+str(j+1))
                pred.append(get_pred(j))

        predictions = np.asarray(pred)

        means = predictions.mean(0)
        stds = predictions.std(0)
        
        
        grad_history = np.load(f"targets/{args.example}/{args.run_name}/cache/grad_history_{setting}.npy")

        if label == 'Mitigation after training':
            setting = "unmitigated_step_no_final_yes"
        if label == 'Mitigation training':
            setting = "realtime_mitigation_step_yes_final_no"
        np.save(arr=means, file=f"targets/{args.example}/{args.run_name}/means_{platform}_{setting}")
        np.save(arr=stds, file=f"targets/{args.example}/{args.run_name}/stds_{platform}_{setting}")

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
    fit_axis.set_xscale(conf["xscale"])
    fit_fig.savefig(f"targets/{args.example}/{args.run_name}/fits_benchmark.pdf", bbox_inches='tight')
    loss_grad_fig.tight_layout()
    loss_grad_fig.savefig(f"targets/{args.example}/{args.run_name}/gradients_analysis.pdf", bbox_inches='tight')
    


# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"targets/{args.example}/cache/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)
