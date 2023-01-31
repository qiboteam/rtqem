import numpy as np
from vqregressor import vqregressor
import matplotlib.pyplot as plt
import argparse


# --------------------- PARSE BEST PARAMS PATH ---------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--best_params_path", 
    default="results/best_params_psr.npy", 
    help="filepath of the best params you want to use for analysing the results", 
    type=str
)

parser.add_argument(
    "--model_info_path",
    default="results/model_info_psr.npy", 
    help="filepath of the file containing some model's info", 
    type=str 
)


# ---------------------- MAIN FUNCTION -----------------------------------------

def main(best_params_path, model_info_path):

    # load best parameters
    best_params = np.load(best_params_path)
    # load model info
    model_info = np.load(model_info_path)

    # define dataset cardinality and number of executions
    parton = 'u'

    nruns = 100

    data = np.loadtxt(f'data/{parton}.dat')
    ndata = len(data)

    x = data.T[0]
    y = data.T[1]

    # initialize vqr with data and best parameters
    VQR = vqregressor(layers=model_info[1], data=x, labels=y)
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


    # plot results
    plt.figure(figsize=(8,6))
    plt.title('Statistics on results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, means, c='purple', alpha=0.7, lw=2, label='Mean values')
    plt.plot(x, y, c='black', lw=2, alpha=0.8, label='Target function')
    plt.fill_between(x, means-stds , means+stds, alpha=0.25, color='purple',
                    label='Confidence belt')
    plt.legend()
    plt.savefig('stat-on-result.png')
    plt.show()

# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)