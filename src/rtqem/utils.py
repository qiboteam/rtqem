from qibo import gates
from qibo.gates.special import FusedGate

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