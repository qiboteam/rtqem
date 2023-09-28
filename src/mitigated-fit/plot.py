import argparse
import random
import json
import os
import scipy
import numpy as np
from prepare_data import prepare_data
import matplotlib.pyplot as plt

platforms = ["sim", "tii1q_b1"]
colors = ["blue", "red"]

parser = argparse.ArgumentParser()
parser.add_argument("example")

parser.add_argument(
    "--platform",
    default="tii1q_b1",
    help="Platform on which we perform predictions.",
    type=str,
)

parser.add_argument(
    "--conf",
    default=None,
    help="filepath of the configuration to use for analysing the results",
    type=str,
)

parser.add_argument(
    "--run_name",
    default='',
    help="Name of the run if data are saved in a sub-folder of the example",
    type=str,
)


def plot(fit_axis, loss_grad_axes, data, means, stds, loss_history, grad_history, color, label):

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
        args.conf if args.conf is not None else f"{args.example}/{args.run_name}/{args.example}.conf"
    )
    with open(conf_file, "r") as f:
        conf = json.load(f)

    platform = conf["platform"]

    files = os.listdir(f"{args.example}/{args.run_name}/cache/")


    data, labels, scaler = prepare_data(
        conf["function"], 
        show_sample=False,
        normalize=conf["normalize_data"], 
        run_name=args.run_name)
    

    if len(np.shape(data)) != 1:
        data1 = data.T[0]
    else:
        data1 = data

    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(5*2/3, 5*(6/8)*2/3))
    fit_axis.plot(data1, labels, c="black", lw=2, alpha=0.8, label="Target function")
    fit_axis.set_title("Statistics on results")
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")

    loss_grad_fig , loss_grad_axes = plt.subplots(2, 1, figsize=(5*2/3, 5*(8/6)*2/3))
    loss_grad_axes[0].set_title('Loss history')
    loss_grad_axes[0].set_ylabel("Loss")
    loss_grad_axes[1].set_title('Grad history')
    loss_grad_axes[1].set_xlabel('Epoch')
    loss_grad_axes[1].set_ylabel('Grad')


    settings, colors, labels = [], [], []

    for f in files:
        if f"best_params_{conf['optimizer']}_noiseless" in f:
            settings.append("noiseless")
            colors.append('green')
            labels.append('Noiseless')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            colors.append('blue')
            labels.append('No mitigation')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            colors.append('orange')
            labels.append('Mitigation after training')
        if f"best_params_{conf['optimizer']}_realtime_mitigation_step_yes_final_yes" in f:
            settings.append("realtime_mitigation_step_yes_final_yes")
            colors.append('red')
            labels.append('Real time mitigation')
        if f"best_params_{conf['optimizer']}_full_mitigation_step_yes_final_yes" in f:
            settings.append("full_mitigation_step_yes_final_yes")
            colors.append('orange')
            labels.append('Full mitigation')

    for setting, color, label in zip(settings, colors, labels):

        loss_history = np.load(f"{args.example}/{args.run_name}/cache/loss_history_{setting}.npy")
        grad_history = np.load(f"{args.example}/{args.run_name}/cache/grad_history_{setting}.npy")
        if label == 'Mitigation after training':
            setting = "unmitigated_step_no_final_yes"
        means = np.load(f"{args.example}/{args.run_name}/means_{platform}_{setting}.npy")
        stds = np.load(f"{args.example}/{args.run_name}/stds_{platform}_{setting}.npy")
    
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

    fit_axis.minorticks_off()
    loss_grad_axes[0].minorticks_off()
    loss_grad_axes[1].minorticks_off()
    fit_axis.legend(loc=3,fontsize="7.5") #uncomment
    fit_axis.set_xscale(conf["xscale"])
    fit_fig.savefig(f"{args.example}/{args.run_name}/fits_benchmark.pdf", bbox_inches='tight')
    loss_grad_axes[0].legend(loc=1,fontsize="7.5") #uncomment
    loss_grad_fig.tight_layout()
    loss_grad_fig.savefig(f"{args.example}/{args.run_name}/gradients_analysis.pdf", bbox_inches='tight')



if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    main(args)