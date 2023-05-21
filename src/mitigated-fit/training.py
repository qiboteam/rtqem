# some useful python package
import numpy as np
import scipy.stats, argparse, json, random
from vqregressor import vqregressor
from qibo.noise import NoiseModel, DepolarizingError
from qibo import gates
from qibo.backends import construct_backend
from qibo.models.error_mitigation import calibration_matrix

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
scaler = lambda x: x
# labeling them
if conf['function'] == 'sinus':
    labels = np.sin(2*data)
elif conf['function'] == 'gamma':
    labels = scipy.stats.gamma.pdf(data, a=2, loc=-1, scale=0.4)
elif conf['function'] == 'gluon':
    scaler = lambda x: np.log(x)
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

if conf['qibolab']:
    backend = construct_backend('qibolab','tii1q_b1')
else:
    backend = None

readout = {}
if conf["mitigation"]['readout'] is not None:
    if conf["mitigation"]['readout'] == 'calibration_matrix':
        cal_m = calibration_matrix(1, backend=backend, noise_model=None, nshots=conf['nshots'])
        np.save('cal_matrix.npy',cal_m)
        readout['calibration_matrix':cal_m]
    elif conf["mitigation"]['readout'] == 'randomized':
        readout['ncircuits':10]
    else:
        raise AssertionError("Invalid readout mitigation method specified.")
    
mit_kwargs = {
    'ZNE': {'noise_levels':np.arange(5), 'insertion_gate':'RX', 'readout':readout},
    'CDR': {'n_training_samples':10, 'readout':readout},
    'vnCDR': {'n_training_samples':10, 'noise_levels':np.arange(3), 'insertion_gate':'RX', 'readout':readout},
    None: {}
}

VQR = vqregressor(
    layers=layers,
    data=data,
    labels=labels,
    nshots=conf['nshots'],
    expectation_from_samples=conf['expectation_from_samples'],
    obs_hardware =conf['obs_hardware'],
    backend = backend,
    noise_model=noise,
    mitigation=conf['mitigation'],
    mit_kwargs=mit_kwargs[conf['mitigation']['method']],
    scaler=scaler
)

# if conf['optimizer'] == 'Adam':
#     # set the training hyper-parameters
#     epochs = conf['epochs']
#     learning_rate = conf['learning_rate']
#     # perform the training
#     history = VQR.gradient_descent(
#         learning_rate=learning_rate, 
#         epochs=epochs, 
#         restart_from_epoch=conf['restart_from_epoch'],
#         batchsize=conf['batchsize'],
#         method='Adam'
#     )
# elif conf['optimizer'] == 'CMA':
#     VQR.cma_optimization()

best_params = np.load('gluon/best_params_Adam.npy',allow_pickle=True)
VQR.params = best_params

VQR.show_predictions(f"{args.example}/predictions_{conf['optimizer']}", save=True)
np.save(f"{args.example}/best_params_{conf['optimizer']}", VQR.params)
