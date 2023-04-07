from qibo.models import Circuit
from qibo import gates
from collections import OrderedDict
import numpy as np


def get_readout_mitigation_matrix(nqubits, backend=None, nshots=1000, p0=None, p1=None):

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend
      backend = GlobalBackend()
    
    matrix = np.zeros((2**nqubits, 2**nqubits))

    states = []
    for i in range(2**nqubits):
        states.append(('{0:0'+str(nqubits)+'b}').format(i))
        
    for i,state in enumerate(states):
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

    nqubits = 3
    nshots = 1000
    p0 = [0.1, 0.2, 0.3]
    p1 = [0.3, 0.1, 0.2]
    mit_m = get_readout_mitigation_matrix(3, nshots=1000, p0=p0, p1=p1)

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
    print(f'> Error Free frequencies:\n {freq}')

    c = Circuit(nqubits)
    c.add(gates.X(0))
    c.add(gates.M(*range(nqubits), p0=p0, p1=p1))

    freq = np.zeros(2**nqubits)
    for k,v in c(nshots=nshots).frequencies().items():
        f = v / nshots
        freq[int(k,2)] = f
    freq = freq.reshape(-1,1)
    print(f'> Noisy frequencies:\n {freq}')
    print(f'> Mitigated frequencies:\n {mit_m @ freq}')

    
