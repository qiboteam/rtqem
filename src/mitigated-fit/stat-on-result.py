import numpy as np
from vqregressor import vqregressor
import matplotlib.pyplot as plt
import scipy.stats, json, os
import argparse, random
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates, set_backend
from qibo.models.error_mitigation import calibration_matrix
from qibo.backends import construct_backend


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
nruns = 50

def plot(fit_axis, loss_grad_axes, data, means, stds, loss_history, grad_history, color):

    global ndata, nruns 

    print(stds)
    
    # plot results
    fit_axis.plot(data, means, c=color, alpha=0.7, lw=2, label="Mean values")
    fit_axis.fill_between(
        data,
        means - stds,
        means + stds,
        alpha=0.25,
        color=color,
        #label="Confidence belt",
    )

    loss_grad_axes[0].plot(loss_history, c=color)
    loss_grad_axes[1].plot(np.sqrt((grad_history*grad_history).sum(-1)), c=color)


    
    
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
        backend = construct_backend("qibojit", platform="numba")
        #backend = construct_backend("numpy")

    # define dataset cardinality and number of executions
    global ndata, nruns

    data = np.linspace(-1, 1, ndata)
    scaler = lambda x: x
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
        idx = np.sort(random.sample(range(len(data)), ndata))
        data = data[idx]
        labels = data.T[1]
        data = data.T[0]

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
        "CDR": {"n_training_samples": 10, "readout": readout, "N_update": 10, "N_mean": 4},
        "vnCDR": {
            "n_training_samples": 10,
            "noise_levels": np.arange(3),
            "insertion_gate": "RX",
            "readout": readout,
        },
        None: {},
    }   

    # plot results
    fit_fig, fit_axis = plt.subplots(figsize=(8, 6))
    fit_axis.plot(data, labels, c="black", lw=2, alpha=0.8, label="Target function")
    fit_axis.set_title("Statistics on results")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")

    loss_grad_fig, loss_grad_axes = plt.subplots(1, 2, figsize=(12,6))
    plt.rcParams['text.usetex'] = True
    loss_grad_axes[0].set_xlabel('Epoch')
    loss_grad_axes[0].set_ylabel("Loss")
    loss_grad_axes[1].set_xlabel('Epoch')
    loss_grad_axes[1].set_ylabel(r'$\|Grad\|$')
    
    files = os.listdir(f"{args.example}/cache/")
    settings, mitigation_settings, colors = [], [], []
    for f in files:
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            mitigation_settings.append({"step":False,"final":False,"method":None,"readout":None})
            colors.append('orange')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_yes_final_yes" in f:
            settings.append("full_mitigation_step_yes_final_yes")
            mitigation_settings.append({"step":True,"final":True,"method":"CDR","readout":"calibration_matrix"})
            colors.append('blue')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_no_final_yes" in f:
            settings.append("full_mitigation_step_no_final_yes")
            mitigation_settings.append({"step":False,"final":True,"method":"CDR","readout":"calibration_matrix"})
            colors.append('green')


    for setting, mitigation, color in zip(settings, mitigation_settings, colors):

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

        for _ in range(nruns):
            predictions.append(VQR.predict_sample())

        predictions = np.asarray(predictions)

        means = []
        stds = []

        for _ in range(ndata):
            means.append(np.mean(predictions.T[_]))
            stds.append(np.std(predictions.T[_]))

        means = np.asarray(means)
        stds = np.asarray(stds)
        
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
            color
        )
    plt.show()

        
    


# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    main(args)
