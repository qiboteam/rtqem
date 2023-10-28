""" Generate data to be fitted """

import json

import numpy as np
import scipy
import matplotlib.pyplot as plt
from qibo.config import log

def prepare_data(example:str, normalize:bool=False, show_sample:bool=False, run_name:str=''):
    """
    Prepare data sample and labels according to example's conf file
    
    Args:
        example: name of the target example
        show_sample: if True, the sampled data and labels are saved as sampled_data.png
    
    Returns sampled_data, labels, scaler
    """

    def funct(z,m,j):
        """
        Generate the target function

        Args:
        - z the variable
        -m the mass (a parameter that should go from 5 to 175 changing from 5 to 5 (5,175,5)
        -j a scaling parameter(0, or 2) when j=1 g(z)=0
        Returns:
        - g: target function """

        # We define the function

        g = (m ** 3 * z ** 2) / (
                    4 * np.pi ** 2 * (-1 + z) ** 4 * np.sqrt((m ** 2 * (1 + 2 * (-1 + z) * z)) / (-1 + z) ** 2)) - (
                        m ** 3 * z ** 2 * (3 / 4 * (1 + j ** (np.log(3) / np.log(2))) ** 2 * m ** 2 * (-1 + z) ** 2 +
                                        m ** 2 * (-1 + z * (2 + z)))) / (
                        8 * np.pi ** 2 * (-1 + z) ** 4 * (1 / 4 * (1 + j ** (np.log(3) / np.log(2))) ** 2 * m ** 2 *
                                                        (-1 + z) ** 2 + m ** 2 * z ** 2) * np.sqrt(
                    1 / 4 * (1 + j ** (np.log(3) / np.log(2))) ** 2 * m ** 2 + (m ** 2 * z ** 2) / (-1 + z) ** 2))
        return g

    if run_name != '':
        conf_file = f"{example}/{run_name}/{example}.conf"
    else:
        conf_file = f"{example}/{example}.conf"

    with open(conf_file, "r") as f:
        conf = json.load(f)

    ndata = conf["ndata"]
    ndim = conf["ndim"]
    function = conf["function"]

    # random data
    data = np.linspace(0.001, 0.999, ndata)
    if ndim != 1:
        # creating matrix of data
        # ndim columns, ndata raws
        data = (np.ones((ndim,1))*data).T
    scaler = lambda x: x
    # labeling them
    if function == "sinus":
        labels = np.sin(2 * data)
    elif function == "hdw_target":
        labels = funct(z=data, m=5, j=2)
        #scaler = lambda x: np.log(x)
    elif function == "gamma":
        labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
    elif function == "gluon":
        scaler = lambda x: np.log(x)
        parton = conf["parton"]
        data = np.loadtxt(f"gluon/data/{parton}.dat")
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

    nlab = len(labels)
    log.info(f"ndata: {nlab}")

    # saving data
    np.save(file=function+"/data.npy", arr=data)
    np.save(file=function+"/labels.npw", arr=labels)

    return data, labels, scaler