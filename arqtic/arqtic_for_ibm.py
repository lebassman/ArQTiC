from .program import Program
import qiskit as qk
from qiskit import Aer, IBMQ, execute

def run_ibm(backend, prog):
    import qiskit as qk
    nqubits = prog.nqubits
    #declare registers
    q_regs = qk.QuantumRegister(nqubits, 'q')
    c_regs = qk.ClassicalRegister(nqubits, 'c')
    #make program into IBM circuit
    ibm_circuit = qk.QuantumCircuit(q_regs, c_regs)
    for gate in prog.gates:
        if (gate.name == "X"):
            ibm_circuit.x(gate.qubits)
        if (gate.name == "Y"):
            ibm_circuit.y(gate.qubits)
        if (gate.name == "Z"):
            ibm_circuit.z(gate.qubits)
        if (gate.name == "H"):
            ibm_circuit.h(gate.qubits)
        if (gate.name == "RZ"):
            ibm_circuit.rz(gate.angles[0], gate.qubits)
        if (gate.name == "CNOT"):
            ibm_circuit.cx(gate.qubits[0], gate.qubits[1])
    #add measurement operators
    ibm_circuit.measure(q_regs, c_regs)
    ibm_circ = qk.transpile(ibm_circuit, backend=backend, optimization_level=1)
    #simulator execution
    #idle simulator run
    result = qk.execute(ibm_circ, backend, shots=1).result().get_counts()
    for spin_str, count in result.items():
        bitstring = [int(s) for s in spin_str]
    #noisy simulator run
    #bitstring = execute(ibm_circ, backend, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates).result()
    return bitstring
