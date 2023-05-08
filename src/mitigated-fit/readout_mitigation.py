from qibo.models import Circuit
from qibo import gates
from collections import OrderedDict
import numpy as np


def get_readout_mitigation_matrix(nqubits, backend=None, nshots=1000, p0=None, p1=None):

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend
      backend = GlobalBackend()
    
    matrix = np.zeros((2**nqubits, 2**nqubits))

    string = '{0:0'+str(nqubits)+'b}'
    for i in range(2**nqubits):
        state = string.format(i)
        circuit = Circuit(nqubits)
        for q,bit in enumerate(state):
            if bit == '1':
                circuit.add(gates.X(q))
        circuit.add(gates.M(*range(nqubits), p0=p0, p1=p1))
        freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()
        column = np.zeros(2**nqubits)
        for key in freq.keys():
            f = freq[key] / nshots 
            column[int(key,2)] = f
        matrix[:,i] = column
    return np.linalg.inv(matrix)

        
if __name__ == '__main__':

    from qibo.backends import GlobalBackend
    backend = GlobalBackend()

    nqubits = 3
    nshots = 1000
    p0 = [0.1, 0.2, 0.3]
    p1 = [0.3, 0.1, 0.2]
    mit_m = get_readout_mitigation_matrix(3, nshots=1000, p0=p0, p1=p1)
    print(mit_m)
    print(mit_m.sum(0))
    
    states = []
    for i in range(2**nqubits):
        states.append(('{0:0'+str(nqubits)+'b}').format(i))

    c = Circuit(nqubits)
    c.add(gates.X(0))
    c.add(gates.M(*range(nqubits)))

    freq = np.zeros(2**nqubits)
    for k,v in c(nshots=nshots).frequencies().items():
        f = v / nshots
        freq[int(k,2)] = f
    freq = freq.reshape(-1,1)
    print(f'> Error Free frequencies:\n {c(nshots=nshots).frequencies()}')

    c = Circuit(nqubits)
    c.add(gates.X(0))
    c.add(gates.M(*range(nqubits), p0=p0, p1=p1))
    
    freq = np.zeros(2**nqubits)
    state = backend.execute_circuit(c, nshots=nshots)
    for k,v in state.frequencies().items():
        f = v / nshots
        freq[int(k,2)] = f
    freq = freq.reshape(-1,1)
    print(state.frequencies())
    mf = mit_m @ freq
    print(mf)
    for i,j in enumerate( mit_m @ freq * nshots):
        state._frequencies[i] = float(j)
    print(state.frequencies())
    #print(f'> Noisy frequencies:\n {freq}')
    #print(f'> Mitigated frequencies:\n {state.frequencies()}')

    
