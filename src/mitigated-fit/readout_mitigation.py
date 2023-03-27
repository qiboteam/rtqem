from qibo.models import Circuit
from qibo import gates
from qibo.backends import NumpyBackend
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
        
    state2idx, state2comb = {}, {}
    np_back = NumpyBackend()
    for c in comb:
        # get the string representing the X gate combinations, e.g. 001, 110, 000, ...
        state = np.zeros(nqubits)
        state[list(c)] = 1
        state = str(state)[1:-1].replace(' ', '').replace('.', '')
        state2comb[state] = c
        # get the order of the states
        # is there a better way?
        circuit = Circuit(nqubits)
        circuit.add([ gates.X(i) for i in c ])
        idx = np_back.execute_circuit(circuit).state().nonzero()
        assert len(idx) == 1
        state2idx[state] = idx[0][0]

    # reorder combinations consistently with qibo ordering
    state2comb = sorted(list(state2comb.items()), key=lambda x: state2idx[x[0]])
    
    for state, c in state2comb:
        circuit = Circuit(nqubits)
        circuit.add([ gates.X(i) for i in c ])
        circuit.add([ gates.M(i) for i in range(nqubits) ])
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()
        column = []
        for k, _ in state2comb:
            f = freq[k] / nshots if k in freq.keys() else 0 
            column.append(f)
        matrix[:,state2idx[state]] = column
    return np.linalg.inv(matrix)

        
if __name__ == '__main__':
    
    from qibo.noise import NoiseModel, DepolarizingError, PauliError
    noise = NoiseModel()
    noise.add(DepolarizingError(lam=0.2), gates.M)

        
    print(get_readout_mitigation_matrix(3, noise_model=noise))
