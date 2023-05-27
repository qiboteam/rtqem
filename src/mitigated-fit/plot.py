import argparse
import random
import json

import scipy
import numpy as np
import matplotlib.pyplot as plt

platforms = ['sim', 'tii1q_b1']
colors = ['blue', 'red']

parser = argparse.ArgumentParser()

parser.add_argument(
    "--example", 
    default="hdw_target", 
    help="Target example.", 
    type=str
)


def main(example):

    with open('{}/{}.conf'.format(example, example), 'r') as f:
        conf = json.loads(f.read())

    # generating target data
    # to have them in the plot
    data = np.linspace(-1, 1, 100)
    if example == 'sinus':
        labels = np.sin(2*data)
    elif example == 'gamma':
        labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
    elif example == 'hdw_target':
        labels = np.sin(2*data) - 0.6*np.cos(4*data)
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
    elif example == 'gluon':
        parton = 'u'
        data = np.loadtxt(f'gluon/data/{parton}.dat')
        idx = random.sample(range(len(data)), conf['ndata'])
        labels = data.T[1][idx]
        data = data.T[0][idx]

    means = []
    stds = []

    # uploading data from all the used platforms
    for p in platforms:
        means.append(np.load(example+f"/means_{p}.npy"))
        stds.append(np.load(example+f"/stds_{p}.npy"))

    x = np.linspace(-1,1,len(means[0]))

    plt.figure(figsize=(8,5))
    plt.plot(data, labels, color='black', alpha=0.7, label='Target')
    for p in range(len(platforms)):
        plt.fill_between(x, means[p]-stds[p], means[p]+stds[p], color=colors[p], alpha=0.3)
        plt.plot(x, means[p],  color=colors[p], alpha=0.7, label=f'Run on {platforms[p]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{example}/bench-platforms.pdf')


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)