import numpy as np
from itertools import product

def get_terms(nqubits):
    obs_list = np.array(list(product(['I','X','Y','Z'],repeat=nqubits)))[1::]
    obs_sign_list = []
    for k, obs in enumerate(obs_list):
        mask = obs != 'I'
        obs_mask = obs[mask]
        obs_sign = np.zeros(len(obs_list))
        for j, obs1 in enumerate(obs_list):
            obs1_mask = obs1[mask]
            mask2 = obs1_mask != 'I'
            index = obs_mask[mask2] != obs1_mask[mask2]
            if list(index).count(True)%2 != 0:
                obs_sign[j] = -2
        obs_sign_list.append(obs_sign)
    return obs_sign_list

def bound_pred(L, nqubits, probs, bit_flip = 0):
    obs_sign_list = get_terms(nqubits)
    qm = 1 - 2*bit_flip
    #px, py, pz = probs
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