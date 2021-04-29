from arqtic.program import Program, Gate
import qiskit as qk
from qiskit import Aer, IBMQ, execute
from arqtic.exceptions import Error

def run_ibm(backend, prog, shots, opt_level=1):
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

def get_ibm_circuit(backend, prog):
    nqubits = prog.nqubits
    #declare registers
    q_regs = qk.QuantumRegister(nqubits, 'q')
    c_regs = qk.ClassicalRegister(nqubits, 'c')
    #make program into IBM circuit
    ibm_circuit = qk.QuantumCircuit(q_regs, c_regs)
    for gate in prog.gates:
        if (gate.name != ""):
            if (gate.name == "X"):
                ibm_circuit.x(gate.qubits)
            elif (gate.name == "Y"):
                ibm_circuit.y(gate.qubits)
            elif (gate.name == "Z"):
                ibm_circuit.z(gate.qubits)
            elif (gate.name == "H"):
                ibm_circuit.h(gate.qubits)
            elif (gate.name == "RZ"):
                ibm_circuit.rz(gate.angles[0], gate.qubits)
            elif (gate.name == "RX"):
                ibm_circuit.rx(gate.angles[0], gate.qubits)
            elif (gate.name == "U3"):
                ibm_circuit.u3(gate.angles[0][0],gate.angles[0][1],gate.angles[0][2], gate.qubits)
            elif (gate.name == "CNOT"):
                ibm_circuit.cx(gate.qubits[0], gate.qubits[1])
            else:
                raise Error(f'Unrecognized gate name: {gate.name}') 
        else: 
            #reverse qubit order for IBM
            rev_q = []
            for q in gate.qubits:
                rev_q.append(nqubits - q - 1)
            #rev_q.append(q)
            #ibm_circuit.unitary(gate.unitary, rev_q)
            ibm_circuit.unitary(gate.unitary, list(reversed(rev_q)))
    
    return ibm_circuit
    
def ibm_circ_to_program(ibm_circ):
    N = ibm_circ.num_qubits
    prog = Program(N)
    #convert IBM circuit into a program
    instr_list=ibm_circ.data
    for instr in instr_list:
        name = instr[0].name
        if name == 'h':
            prog.add_gate(Gate([instr[1][0].index], 'H'))
        elif name == 'sx':
            prog.add_gate(Gate([instr[1][0].index], 'SX'))
        elif name == "rx":
            prog.add_gate(Gate([instr[1][0].index],'RX', angles=[instr[0].params[0]]))
        elif name == "ry":
            prog.add_gate(Gate([instr[1][0].index],'RY', angles=[instr[0].params[0]]))
        elif name == "rz":
            prog.add_gate(Gate([instr[1][0].index],'RZ', angles=[instr[0].params[0]]))
        elif name == "cx":
            prog.add_gate(Gate([instr[1][0].index, instr[1][1].index], 'CNOT'))
        elif name == "u3":
            prog.add_gate(Gate([instr[1][0].index],'U3', angles=[instr[0].params]))
        else: 
            raise Error(f'Unrecognized instruction: {name}')  
    return prog
    
def add_prog_to_ibm_circuit(backend, prog, ibm_circuit, transpile=True, opt_level=1, basis_gates = ['cx', 'u1', 'u2', 'u3']):
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

