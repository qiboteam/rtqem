from qibo.models import Circuit
from qibo import gates
from itertools import combinations
import numpy as np


def get_readout_mitigation_matrix(nqubits, backend=None, nshots=1000, noise_model=None):

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend
      backend = GlobalBackend()
    
    matrix = np.zeros((2**nqubits, 2**nqubits))
      
    comb = []
    for i in range(nqubits + 1):
        comb += list(combinations(range(nqubits), i)) #check that the ordering is consistent with qibo
        
    ordered_keys = []
    for c in comb:
        state = np.zeros(nqubits)
        state[list(c)] = 1
        state = str(state)[1:-1].replace(' ', '').replace('.', '')
        ordered_keys.append(state)
        
    for n,c in enumerate(comb):
        circuit = Circuit(nqubits)
        circuit.add([ gates.X(i) for i in c ])
        circuit.add([ gates.M(i) for i in range(nqubits) ])
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()
        column = []
        for k in ordered_keys:
            f = freq[k] / nshots if k in freq.keys() else 0 
            column.append(f)
        matrix[:,n] = column
    print(matrix)
    return np.linalg.inv(matrix)

        
if __name__ == '__main__':
    
    from qibo.noise import NoiseModel, DepolarizingError
    noise = NoiseModel()
    noise.add(DepolarizingError(lam=0.5), gates.X)
        
    print(get_readout_mitigation_matrix(3, noise_model=noise))
