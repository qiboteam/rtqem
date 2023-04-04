from qibo.models import Circuit
from qibo import gates
from collections import OrderedDict
import numpy as np


def get_readout_mitigation_matrix(nqubits, backend=None, nshots=1000, noise_model=None):

    if backend is None:  # pragma: no cover
      from qibo.backends import GlobalBackend
      backend = GlobalBackend()
    
    matrix = np.zeros((2**nqubits, 2**nqubits))

    states = []
    for i in range(2**nqubits):
        states.append(('{0:0'+str(nqubits)+'b}').format(i))

    if noise_model is not None:
        # random bitflips for testing
        p0 = list(0.2*np.random.rand(nqubits))
    else:
        p0 = list(np.zeros(nqubits))

    for i,state in enumerate(states):
        print(state)
        circuit = Circuit(nqubits)
        for q,bit in enumerate(state):
            if bit == '1':
                circuit.add(gates.X(q))
        circuit.add(gates.M(*range(nqubits), p0=p0))
        freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()
        print(freq)
        column = []
        for key in states:
            f = freq[key] / nshots if key in freq.keys() else 0 
            column.append(f)
        print(column)
        matrix[:,i] = column
    print(matrix)
    return np.linalg.inv(matrix)

        
if __name__ == '__main__':
    
    from qibo.noise import NoiseModel
    noise = NoiseModel()
    par = {
        "t1" : (250*1e-06, 240*1e-06),
        "t2" : (150*1e-06, 160*1e-06),
        "gate_time" : (200*1e-9, 400*1e-9),
        "excited_population" : 0,
        "depolarizing_error" : (4.000e-4, 1.500e-4),
        "bitflips_error" : ([0.022, 0.015], [0.034, 0.041]),
        "idle_qubits" : 1
    }
    noise.composite(par)
    
        
    print(get_readout_mitigation_matrix(3, noise_model=noise))
