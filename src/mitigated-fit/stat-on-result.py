import numpy as np
from vqregressor import vqregressor
import matplotlib.pyplot as plt
import scipy.stats, json
import argparse, random


# --------------------- PARSE BEST PARAMS PATH ---------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('example')

parser.add_argument(
    "--best_params_path", 
    default=None, 
    help="filepath of the best params you want to use for analysing the results", 
    type=str
)

parser.add_argument(
    "--conf",
    default=None,
    help="filepath of the configuration to use for analysing the results", 
    type=str
)

# ---------------------- MAIN FUNCTION -----------------------------------------

def main(args):

    conf_file = args.conf if args.conf is not None else f'{args.example}/{args.example}.conf'
    with open(conf_file, 'r') as f:
        conf = json.load(f)

    # load best parameters
    if args.best_params_path is not None:
        best_params = np.load(args.best_params_path)
    else:
        best_params = np.load(f"{args.example}/best_params_{conf['optimizer']}.npy")
        
    # define dataset cardinality and number of executions
    ndata = 100
    nruns = 100

    data = np.linspace(-1, 1, ndata)
    scaler = lambda x: x
    if conf['function'] == 'sinus':
        labels = np.sin(2*data)
    elif conf['function'] == 'gamma':
        labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
    elif conf['function'] == 'gluon':
        scaler = lambda x: np.log(x)
        parton = conf['parton']
        data = np.loadtxt(f'gluon/data/{parton}.dat')
        labels = data.T[1]
        data = data.T[0]

    # noise model
    if conf['noise']:
        noise = NoiseModel()
        noise.add(DepolarizingError(lam=0.25), gates.RZ)
    else:
        noise = None

    mit_kwargs = {
        'ZNE': {'noise_levels':np.arange(5), 'insertion_gate':'RX'},
        'CDR': {'n_training_samples':10},
        'vnCDR': {'n_training_samples':10, 'noise_levels':np.arange(3), 'insertion_gate':'RX'},
        None: {}
    }
    
    # initialize vqr with data and best parameters
    VQR = vqregressor(
        layers=conf['nlayers'],
        data=data,
        labels=labels,
        nshots=conf['nshots'],
        expectation_from_samples=conf['expectation_from_samples'],
        noise_model=noise,
        mitigation=conf['mitigation'],
        mit_kwargs=mit_kwargs[conf['mitigation']],
        scaler=scaler
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


    # plot results
    plt.figure(figsize=(8,6))
    plt.title('Statistics on results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(data, means, c='purple', alpha=0.7, lw=2, label='Mean values')
    plt.plot(data, labels, c='black', lw=2, alpha=0.8, label='Target function')
    plt.fill_between(data, means-stds , means+stds, alpha=0.25, color='purple',
                    label='Confidence belt')
    plt.legend()
    plt.savefig('stat-on-result.png')
    plt.show()

# ---------------------- EXECUTE MAIN ------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    if args.example[-1] == '/':
        args.example = args.example[:-1]
    main(args)
