# some useful python package
import numpy as np
import scipy.stats, argparse, json, random
from vqregressor import vqregressor
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates

parser = argparse.ArgumentParser(description='Training the vqregressor')
parser.add_argument('example')

args = parser.parse_args()

if args.example[-1] == '/':
    args.example = args.example[:-1]

with open('{}/{}.conf'.format(args.example, args.example), 'r') as f:
    conf = json.loads(f.read())

# model definition
nqubits = conf['nqubits']
layers = conf['nlayers']
ndata = conf['ndata']

# random data
data = np.random.uniform(-1, 1, ndata)
scaler = lambda x: np.log(x)
# labeling them
if conf['function'] == 'sinus':
    labels = np.sin(2*data)
elif conf['function'] == 'gamma':
    labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
elif conf['function'] == 'gluon':
    scaler = lambda x: x#np.log(x)
    parton = conf['parton']
    data = np.loadtxt(f'gluon/data/{parton}.dat')
    idx = random.sample(range(len(data)), ndata)
    labels = data.T[1][idx]
    data = data.T[0][idx]

# noise model
if conf['noise']:
    noise = NoiseModel()
    noise.add(DepolarizingError(lam=0.1), gates.RX)
else:
    noise = None

mit_kwargs = {
    'ZNE': {'noise_levels':np.arange(5), 'insertion_gate':'RX'},
    'CDR': {'n_training_samples':10},
    'vnCDR': {'n_training_samples':10, 'noise_levels':np.arange(3), 'insertion_gate':'RX'},
    None: {}
}

VQR = vqregressor(
    layers=layers,
    data=data,
    labels=labels,
    min_shots=10,
    nshots=conf['nshots'],
    expectation_from_samples=conf['expectation_from_samples'],
    obs_hardware =conf['obs_hardware'], 
    noise_model=noise,
    mitigation=conf['mitigation'],
    mit_kwargs=mit_kwargs[conf['mitigation']['method']],
    scaler=scaler
)

if conf['optimizer'] == 'Adam' or conf['optimizer'] == 'Rosalin' or conf['optimizer'] == 'shoals':
    # set the training hyper-parameters
    epochs = conf['epochs']
    learning_rate = conf['learning_rate']
    b = conf['b']
    mu = conf['mu']
    p = conf['p']
    gamma = conf['gamma']
    c = conf['c']
    eps_f = conf['eps_f']
    learning_rate_max = conf['learning_rate_max'] 
    # perform the training
    history = VQR.gradient_descent(
        learning_rate=learning_rate,
        learning_rate_max=learning_rate_max,  
        p=p,
        gamma=gamma,
        eps_f=eps_f,
        c=c,
        b=b,
        mu=mu,
        epochs=epochs, 
        restart_from_epoch=conf['restart_from_epoch'],
        batchsize=conf['batchsize'],
        method=conf['optimizer']
    )
elif conf['optimizer'] == 'CMA':
    VQR.cma_optimization()


VQR.show_predictions(f"{args.example}/predictions_{conf['optimizer']}", save=True)
np.save(f"{args.example}/best_params_{conf['optimizer']}", VQR.params)
