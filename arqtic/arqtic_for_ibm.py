from arqtic.program import Program
import qiskit as qk
from qiskit import Aer, IBMQ, execute

def run_ibm(backend, prog, shots, opt_level=1):
    nqubits = prog.nqubits
    #declare registers
    q_regs = qk.QuantumRegister(nqubits, 'q')
    c_regs = qk.ClassicalRegister(nqubits, 'c')
    #make program into IBM circuit
    ibm_circuit = qk.QuantumCircuit(q_regs, c_regs)
    for gate in prog.gates:
        #print(gate.name)
        #print(gate.qubits)
        #print(gate.angles)
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
        if (gate.name == "RX"):
            ibm_circuit.rx(gate.angles[0], gate.qubits)
        if (gate.name == "CNOT"):
            ibm_circuit.cx(gate.qubits[0], gate.qubits[1])
    #add measurement operators
    ibm_circuit.measure(q_regs, c_regs)
    ibm_circ = qk.transpile(ibm_circuit, backend=backend, optimization_level=opt_level)
    #simulator execution
    #idle simulator run
    result = qk.execute(ibm_circ, backend, shots=shots).result().get_counts()
    results = []
    for spin_str, count in result.items():
        results.append([])
        results[-1].append([int(s) for s in spin_str])
        results[-1].append(count)
    #noisy simulator run
    #bitstring = execute(ibm_circ, backend, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates).result()
    return results

def get_ibm_circuit(backend, prog, transpile=True, opt_level=1, basis_gates = ['cx', 'u1', 'u2', 'u3']):
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
        if (gate.name == "RX"):
            ibm_circuit.rx(gate.angles[0], gate.qubits)
        if (gate.name == "CNOT"):
            ibm_circuit.cx(gate.qubits[0], gate.qubits[1])
    #add measurement operators
    if (transpile):
        ibm_circ = qk.transpile(ibm_circuit, backend=backend, optimization_level=opt_level, basis_gates=basis_gates)
        return ibm_circ
    else:
        return ibm_circuit
