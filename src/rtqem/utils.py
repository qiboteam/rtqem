from qibo import gates
from qibo.gates.special import FusedGate
from qibo.backends import NumpyBackend
from qibo.models.circuit import Circuit
from qibo.result import MeasurementOutcomes
from qiboconnection import API
from collections import Counter
import numpy as np


class FusedGate1(FusedGate):
    @classmethod
    def from_gate(cls, gate):
        fgate = cls(*gate.qubits)
        fgate.append(gate)
        if isinstance(gate, (gates.M, gates.SpecialGate, gates.I)):
            # special gates do not participate in fusion
            fgate.marked = True
        return fgate


def to_fused(queue):
    """Transform all gates in queue to :class:`qibo.gates.FusedGate`."""
    last_gate = {}
    queue_new = queue.__class__(queue.nqubits)
    for gate in queue:
        fgate = FusedGate1.from_gate(gate)
        if isinstance(gate, gates.SpecialGate):
            fgate.qubit_set = set(range(queue.nqubits))
            fgate.init_args = sorted(fgate.qubit_set)
            fgate.target_qubits = tuple(fgate.init_args)

        for q in fgate.qubits:
            if q in last_gate:
                neighbor = last_gate.get(q)
                fgate.left_neighbors[q] = neighbor
                neighbor.right_neighbors[q] = fgate
            last_gate[q] = fgate
        queue_new.append(fgate)
    return queue_new
    
def fuse(circuit, max_qubits=2):
    """Creates an equivalent circuit by fusing gates for increased
    simulation performance.

    Args:
        max_qubits (int): Maximum number of qubits in the fused gates.

    Returns:
        A :class:`qibo.core.circuit.Circuit` object containing
        :class:`qibo.gates.FusedGate` gates, each of which
        corresponds to a group of some original gates.
        For more details on the fusion algorithm we refer to the
        :ref:`Circuit fusion <circuit-fusion>` section.

    Example:
        .. testcode::

            from qibo import gates, models
            c = models.Circuit(2)
            c.add([gates.H(0), gates.H(1)])
            c.add(gates.CNOT(0, 1))
            c.add([gates.Y(0), gates.Y(1)])
            # create circuit with fused gates
            fused_c = c.fuse()
            # now ``fused_c`` contains a single ``FusedGate`` that is
            # equivalent to applying the five original gates
    """
    queue = to_fused(circuit.queue)
    for gate in queue:
        if not gate.marked:
            for q in gate.qubits:
                # fuse nearest neighbors forth in time
                neighbor = gate.right_neighbors.get(q)
                if gate.can_fuse(neighbor, max_qubits):
                    gate.fuse(neighbor)
                # fuse nearest neighbors back in time
                neighbor = gate.left_neighbors.get(q)
                if gate.can_fuse(neighbor, max_qubits):
                    neighbor.fuse(gate)
    # create a circuit and assign the new queue
    circuit_new = circuit._shallow_copy()
    circuit_new.queue = from_fused(queue)
    return circuit_new


def from_fused(queue):
    """Create queue from fused circuit.

    Create the fused circuit queue by removing gates that have been
    fused to others.
    """
    queue_new = queue.__class__(queue.nqubits)
    for gate in queue:
        if not gate.marked:
            if len(gate.gates) == 1:
                # replace ``FusedGate``s that contain only one gate
                # by this gate for efficiency
                queue_new.append(gate.gates[0])
            else:
                queue_new.append(gate)
        elif isinstance(gate.gates[0], (gates.SpecialGate, gates.M, gates.I)):
            # special gates are marked by default so we need
            # to add them manually
            queue_new.append(gate.gates[0])
    return queue_new

class QuantumSpain(NumpyBackend):
    def __init__(self, configuration, device_id, nqubits, qubit_map=None):
        super().__init__()
        self.name = "QuantumSpain"
        self.platform = API(configuration = configuration)
        self.platform.select_device_id(device_id=device_id)
        self.nqubits = nqubits
        self.qubit_map = qubit_map
    def transpile_circ(self, circuit, qubit_map=None):
        if qubit_map == None:
            qubit_map = list(range(circuit.nqubits))
        self.qubit_map = qubit_map
        circuit = fuse(circuit, max_qubits=1)
        from qibolab.transpilers.unitary_decompositions import u3_decomposition
        new_c = Circuit(self.nqubits, density_matrix=True)
        for gate in circuit.queue:
            qubits = [self.qubit_map[j] for j in gate.qubits]
            if isinstance(gate, gates.M):
                new_gate = gates.M(*tuple(qubits), **gate.init_kwargs)
                new_gate.result = gate.result
                new_c.add(new_gate)
            elif isinstance(gate, gates.I):
                new_c.add(gate.__class__(*tuple(qubits), **gate.init_kwargs))
            else:
                matrix = gate.matrix()
                theta, phi, lamb = u3_decomposition(matrix)
                new_c.add([gates.RZ(*tuple(qubits),lamb),gates.RX(*tuple(qubits),np.pi/2),gates.RZ(*tuple(qubits),theta+np.pi),gates.RX(*tuple(qubits),np.pi/2),gates.RZ(*tuple(qubits),phi+np.pi)])#gates.U3(*tuple(qubits), *u3_decomposition(matrix)))
        return new_c
    def execute_circuit(self, circuits, nshots=1000):
        if isinstance(circuits, list) is False:
            circuits = [circuits]
        for k in range(len(circuits)):
            circuits[k] = self.transpile_circ(circuits[k], self.qubit_map)
        results = self.platform.execute_and_return_results(circuits, nshots=nshots, interval=10)[0]
        result_list = []
        for j, result in enumerate(results):
            probs = result['probabilities']
            counts = Counter()
            for key in probs:
                counts[int(key,2)] = int(probs[key]*nshots)
            result = MeasurementOutcomes(circuits[j].measurements, self, nshots=nshots)
            result._frequencies = counts
            result_list.append(result)
        if len(result_list) == 1:
            return result_list[0]
        return result_list