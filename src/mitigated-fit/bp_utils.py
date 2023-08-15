import numpy as np
from itertools import product

def get_terms(nqubits):
    num_terms = 3
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
    tr_distance = 1 #2*(1-1/2**nqubits)
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