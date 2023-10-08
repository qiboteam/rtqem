import argparse
import random
import json
import os
import numpy as np
from prepare_data import prepare_data
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')

platforms = ["sim", "tii1q_b1"]
colors = ["blue", "red"]

parser = argparse.ArgumentParser()
parser.add_argument("example")

parser.add_argument(
    "--platform",
    default="sim",
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

parser.add_argument(
    "--legends",
    default="true",
    help="Set true if legends are desired in the plots, false if not.",
    type=str,
)

parser.add_argument(
    "--linewidth",
    default=0.5,
    help="Manuscript linewidth occupied by the plot.",
    type=float,
)

def plot(fit_axis, loss_axis, grad_axis, data, means, stds, loss_history, grad_history, color, label):

    global ndata, nruns
    
    if len(np.shape(data)) != 1:
        data = data.T[0]

    # plot results
    fit_axis.plot(data, means, c=color, alpha=0.8, lw=1.5, label=label)
    fit_axis.fill_between(
        data,
        means - stds,
        means + stds,
        alpha=0.3,
        hatch="//",
        color=color,
    )
    if label != "Mitigation after training":
        # plot loss history
        loss_axis.plot(loss_history, c=color, lw=1.5, alpha=0.8, label=label)
        loss_axis.set_yscale('log')
        # plot grad history
        grad_axis.set_yscale('log')
        grad_axis.plot(
            np.mean(np.sqrt((grad_history*grad_history)),axis=-1), 
            c=color,
            lw=1.5,
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
    width = args.linewidth


    data, labels, _ = prepare_data(
        conf["function"], 
        show_sample=False,
        normalize=conf["normalize_data"], 
        run_name=args.run_name)
    

    if len(np.shape(data)) != 1:
        data1 = data.T[0]
    else:
        data1 = data

    fit_fig , fit_axis = plt.subplots(1, 1, figsize=(8 * width, 8 * (6/8) * width))
    fit_axis.plot(data1, labels, c="black", lw=1.5, alpha=0.8, label="Target function")
    fit_axis.set_title(fr"Simulated fit", fontsize=12)
    fit_axis.set_xlabel("x")
    fit_axis.set_ylabel("y")

    loss_fig , loss_axis = plt.subplots(1, 1, figsize=(8 * width, 8 * (6/8) * width))
    loss_axis.set_title(fr'Loss history', fontsize=12)
    loss_axis.set_xlabel('Epoch')
    loss_axis.set_ylabel("Loss")

    grad_fig , grad_axis = plt.subplots(1, 1, figsize=(8 * width, 8 * (6/8) * width))
    grad_axis.set_title(fr'Grad history', fontsize=12)
    grad_axis.set_xlabel('Epoch')
    grad_axis.set_ylabel('Grad')


    settings, colors, labels = [], [], []

    for f in files:
        if f"best_params_{conf['optimizer']}_noiseless" in f:
            settings.append("noiseless")
            colors.append('#44c24d')
            labels.append('Noiseless')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            colors.append('#4287f5')
            labels.append('No mitigation')
        if f"best_params_{conf['optimizer']}_unmitigated" in f:
            settings.append("unmitigated_step_no_final_no")
            colors.append('orange')
            labels.append('Mitigation after training')
        if f"best_params_{conf['optimizer']}_realtime_mitigation_step_yes_final_yes" in f:
            settings.append("realtime_mitigation_step_yes_final_yes")
            colors.append('#f54242')
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
            loss_axis,
            grad_axis,
            data,
            means,
            stds,
            loss_history,
            grad_history,
            color,
            label
        )

    fit_axis.minorticks_off()
    loss_axis.minorticks_off()
    grad_axis.minorticks_off()

    if args.legends == "true":
        fit_axis.legend(loc=3, fontsize=7) 
        loss_axis.legend(loc=1, fontsize=7)

    fit_axis.set_xscale(conf["xscale"])
    fit_fig.savefig(f"{args.example}/{args.run_name}/{args.run_name}.pdf", bbox_inches='tight', dpi=200)
    loss_fig.savefig(f"{args.example}/{args.run_name}/{args.run_name}_loss.pdf", bbox_inches='tight', dpi=200)
    grad_fig.savefig(f"{args.example}/{args.run_name}/{args.run_name}_grad.pdf", bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    args = parser.parse_args()  
    if args.example[-1] == "/":
        args.example = args.example[:-1]
    cache_dir = f"{args.example}/cache/"
    main(args)