from functools import reduce
from itertools import product

import numpy as np

from qibo.noise import NoiseModel, PauliError, ReadoutError, DepolarizingError
from qibo import gates

def get_terms(nqubits):
    num_terms = 3#4**nqubits-1
    terms = [-2]*num_terms
    terms_list = []
    for k in range(num_terms):
        term = terms.copy()
        term[k] = 0
        terms_list.append(term)
    return terms_list


def bound_pred(L, nqubits, probs, bit_flip = 0):
    obs_sign_list = get_terms(nqubits)
    qm = 1 - 2*bit_flip
    tr_distance = 2*(1-1/2**nqubits)
    N0 = 1
    w_inf = 1
    qs = []
    for term in obs_sign_list:
        qs.append(1+np.dot(term,probs))
    q = np.sqrt(np.max(np.abs(qs)))
    G = N0*w_inf*q**(2*L+2)
    return G*tr_distance*qm**nqubits

def bound_grad(L, nqubits, probs, bit_flip = 0):
    obs_sign_list = get_terms(nqubits)
    qm = 1 - 2*bit_flip
    h_lm = 1
    N0 = 1
    w_inf = 1
    qs = []
    for term in obs_sign_list:
        qs.append(1+np.dot(term,probs))
    q = np.sqrt(np.max(np.abs(qs)))
    F = np.sqrt(8*np.log(2))*N0*h_lm*w_inf*np.sqrt(nqubits)*q**(L+1)
    return F*qm**nqubits


def generate_noise_model(qm, nqubits, noise_magnitude):
    """Generate noise model to perform noisy simulations.
    
    Args:
        qm (float): readout matrix error magnitude.
        nqubits (int): number of qubits.
        noise_magnitude (float): noise magnitude which affects Pauli gates.
        
    Returns: `qibo.model.noise`
    """

    paulis = list(product(["I", "X", "Y", "Z"], repeat=1))[1:]
    #np.random.seed(123)
    probabilities = noise_magnitude#np.random.rand(len(paulis))*noise_magnitude#np.repeat(noise_magnitude, repeats=len(paulis)) 
    single_readout_matrix = np.array([[1-qm,qm],[qm,1-qm]])
    pauli_noise = PauliError(list(zip(paulis, probabilities)))
    #depol_noise = DepolarizingError(noise_magnitude*4**nqubits)
    readout_noise = ReadoutError(single_readout_matrix)

    noise = NoiseModel()
    #noise.add(depol_noise, gates.I)
    noise.add(pauli_noise, gates.I)
    noise.add(readout_noise, gates.M)

    return noise