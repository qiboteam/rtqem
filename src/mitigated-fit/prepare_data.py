""" Generate data to be fitted """

import json

import numpy as np
import scipy
import matplotlib.pyplot as plt

def prepare_data(example:str, normalize:bool=False, show_sample:bool=False, run_name:str=''):
    """
    Prepare data sample and labels according to example's conf file
    
    Args:
        example: name of the target example
        show_sample: if True, the sampled data and labels are saved as sampled_data.png
    
    Returns sampled_data, labels, scaler
    """

    if run_name != '':
        conf_file = f"targets/{example}/{run_name}/{example}.conf"
    else:
        conf_file = f"targets/{example}/{example}.conf"

    with open(conf_file, "r") as f:
        conf = json.load(f)

    ndata = conf["ndata"]
    ndim = conf["ndim"]
    function = conf["function"]

    # random data
    data = np.linspace(-1, 1, ndata)
    if ndim != 1:
        # creating matrix of data
        # ndim columns, ndata raws
        data = (np.ones((ndim,1))*data).T
    scaler = lambda x: x
    # labeling them
    if function == "sinus":
        labels = np.sin(2 * data)
    elif function == "hdw_target":
        labels = np.exp(-data) * np.cos(3 * data) * 0.3
    elif function == "gamma":
        labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
    elif function == "uquark":
        scaler = lambda x: np.log(x)
        data = np.loadtxt(f"uquark/data/u.dat")
        idx = np.round(np.linspace(0,len(data)-1,ndata)).astype(int)
        labels = data.T[1][idx]
        data = data.T[0][idx]
    elif function == "cosnd":
        data = np.linspace(-1, 1, ndata)
        data = (np.ones((ndim,1))*data).T
        thetas = np.linspace(0.5, 2.5, ndim)
        labels = np.zeros(ndata)

        for dim in range(ndim):
            contribute = np.cos(thetas[dim]*data.T[dim])**[dim+1] + ((-1)**dim)*thetas[dim]*data.T[dim]
            labels += contribute
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))

    if normalize:
        # normalize labels to be in [0,1]
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))

    # print in case you want to have a look to the data     
    if show_sample:
        if ndim != 1:
            xarr = data.T[0]
        else:
            xarr = data

        plt.title('Sampled data')
        plt.plot(xarr, labels, color="blue", alpha=0.7, lw=2)
        plt.xlabel(r"x_0")
        plt.ylabel("y")
        plt.savefig("sampled_data.png")

    # saving data
    np.save(file=function+"/data.npy", arr=data)
    np.save(file=function+"/labels.npw", arr=labels)

    return data, labels, scaler