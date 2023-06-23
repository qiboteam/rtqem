import numpy as np

def bound_pred(probs, L, nqubits):
    px, py, pz = probs
    tr_distance = 2*(1-1/2**nqubits)
    N0 = 1
    w_inf = 1
    qx = 1-2*py-2*pz
    qy = 1-2*px-2*pz
    qz = 1-2*px-2*py
    q = np.sqrt(np.max(np.abs([qx,qy,qz])))
    G = N0*w_inf*q**(2*L+2)
    return G*tr_distance

def bound_grad(probs, L, nqubits):
    px, py, pz = probs
    h_lm = 1
    N0 = 1
    w_inf = 1
    qx = 1-2*py-2*pz
    qy = 1-2*px-2*pz
    qz = 1-2*px-2*py
    q = np.sqrt(np.max(np.abs([qx,qy,qz])))
    F = np.sqrt(8*np.log(2))*N0*h_lm*w_inf*np.sqrt(nqubits)*q**(L+1)
    return F