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

# ---------------------- MAIN FUNCTION -----------------------------------------

def main(best_params_path):

    # load best parameters
    best_params = np.load(best_params_path)

    # define dataset cardinality and number of executions
    ndata = 100
    nruns = 100

    data = np.linspace(-1, 1, ndata)
    labels = np.sin(2*data)

    # initialize vqr with data and best parameters
    VQR = vqregressor(layers=1, data=data, labels=labels)
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
    plt.plot(data, means, c='purple', alpha=0.7, lw=2, label='Mean values')
    plt.fill_between(data, means-stds , means+stds, alpha=0.25, color='purple',
                    label='Confidence belt')
    plt.legend()
    plt.savefig('stat-on-result.png')
    plt.show()

# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)