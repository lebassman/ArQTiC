from pyquil.api import get_qc
from pyquil.quil import Program
from pyquil.paulis import sX, sY, PauliTerm, exponential_map
from pyquil.gates import RZ, Z, H, MEASURE, CNOT, RESET
from pyquil.quilbase import (DefGate, Gate, Measurement, AbstractInstruction, Qubit, Declare, Reset)
import numpy as np
from smart_compile import smart_compile 
 
# Physical constants, global variable
H_BAR = 0.658212    # eV*fs
Jz = 0.01183898    # eV, coupling coeff; Jz<0 is antiferromagenetic, Jz>0 is ferromagnetic
FREQ = 0.0048       # 1/fs, frequency of MoSe2 phonon
Jx = 2.0*0.01183898
 
def setup_forest_objects():
    """Set up quantum virtual machine."""
    qc = get_qc("Aspen-7-5Q-B", as_qvm=True)    ##Choose a specific lattice, and whether to run as a qvm or not!
    #qc = get_qc("4q-qvm") #Generic qvm with all-to-all connections; change '2' to any number of qubits you wish to use
    return qc
 
def evolution_circuit(nqubits, delta_t, total_time, w_ph):
    """
        Define circuit for evolution of wavefunction, i.e.,
        H(t) = - Jz * sum_{i=1}^{N-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
        - e_ph * cos(w_ph * t) * sum_{i=1}^{N}(sigma_{x}^{i})
         
        Args:
        - qubits: list of qubits in system
        - delta_t: unit of propagation time
        - total_time: total time to evolve system
        - w_ph: angular frequency of phonon
         
        Return:
        - pyquil.Program
        """
    #define e_ph stregnth 
    e_ph = 1.0*Jz
     
    #instantiate program object for return
    p = Program(RESET())

    # decalre time e_ph as a parameter
    # declare memory for read out
    ro = p.declare('ro', memory_type='BIT', memory_size=nqubits)

    # determine number of time steps to get to total time
    prop_steps = int(total_time / delta_t)
     
    # instantiate program object for the propagator to which
    # we add terms of Hamiltonian piece by piece
    for step in range(0,prop_steps):
        t = (step + 0.5) * delta_t
        propagator_t = Program()
        instr_set1 = []
        instr_set2 = []
        instr_set3 = []
        theta_x = -e_ph * np.cos(w_ph * t)
        #make Z coupling terms
        coupling_termsZ = []
        for i in range(0, nqubits-1):
            coupling_termsZ.append(PauliTerm("Z", i, -Jz)*PauliTerm("Z", i+1))

        #make X coupling terms
        coupling_termsX = []
        for i in range(0, nqubits-1):
            coupling_termsX.append(PauliTerm("X", i, -Jx)*PauliTerm("X", i+1))

        #make Y coupling terms
        coupling_termsY = []
        for i in range(0, nqubits-1):
            coupling_termsY.append(PauliTerm("Y", i, -Jx)*PauliTerm("Y", i+1))

        #make transverse magnetization terms of Hamiltonian
        Hx = []
        for i in range(0,nqubits):
            Hx.append(PauliTerm("X", i, theta_x))

        for j in range(0, nqubits):
            instr_set1.append(exponential_map(Hx[j])(delta_t/H_BAR))
        for j in range(0, nqubits-1):
            #construct j=even product
            if (j % 2 == 0) : 
                instr_set2.append(exponential_map(coupling_termsX[j])(delta_t/H_BAR))
                instr_set2.append(exponential_map(coupling_termsY[j])(delta_t/H_BAR))
                instr_set2.append(exponential_map(coupling_termsZ[j])(delta_t/H_BAR))
            #construct j=odd product
            else: 
                instr_set3.append(exponential_map(coupling_termsX[j])(delta_t/H_BAR))
                instr_set3.append(exponential_map(coupling_termsY[j])(delta_t/H_BAR))
                instr_set3.append(exponential_map(coupling_termsZ[j])(delta_t/H_BAR))

        # create propagator
        propagator_t.inst(instr_set1, instr_set2, instr_set3)
        p.inst(propagator_t)

    # add measurement operators to each qubit
    for i in range(0,nqubits):
        p.inst(MEASURE(i, ro[i]))
 
    # return program
    return p

def map_qubits(program, qubit_mapping):
    result = []
    for instr in program:
        if isinstance(instr, Gate):
            remapped_qubits = [qubit_mapping[q] for q in instr.qubits]
            gate = Gate(instr.name, instr.params, remapped_qubits)
            gate.modifiers = instr.modifiers
            result.append(gate)
        elif isinstance(instr, Measurement):
            result.append(Measurement(qubit_mapping[instr.qubit], instr.classical_reg)) 
        else:
            result.append(instr)
    new_program = program.copy()
    new_program._instructions = result

    return new_program


if __name__=="__main__":
    import time
     
    # set up the Forest object
    qc = setup_forest_objects()

    #define desired number of qubits in system
    #nqubits = len(qubits)
    nqubits = 4

    #define time-step and total time
    #should have total_t divisible by delta_t
    delta_t = 3     # fs
    total_t = 60   # fs
    num_steps = int(total_t / delta_t) + 1

    #define number of trials per circuit
    trials = 10000

    # define the terms of the Hamiltonian
    w_ph = 2.0 * np.pi * FREQ   # 1/fs
     
    # loop over time steps
    with open("test.npy", "wb") as f:
        np.save(f, (trials, nqubits, num_steps, 1))
        for i in range(0, num_steps):
            start_step = time.time()
            #compute total time to evolve wavefunction under for this step of loop
            evolution_time = delta_t * i
            #create high-level program assuming qubits labeled 0,1,2,...,nqubits
            program = evolution_circuit(nqubits, delta_t, evolution_time, w_ph)

            #Proprietary compilation:
            #program = smart_compile(program, nqubits)

            #address qubits to specific lattice being used
            mapped_program = map_qubits(program, qubit_mapping={
                           Qubit(0): Qubit(10),
                           Qubit(1): Qubit(11),
                           Qubit(2): Qubit(12), 
                           Qubit(3): Qubit(13),})
            mapped_program.wrap_in_numshots_loop(trials)

            #Rigetti compilation
            mapped_program = qc.compiler.quil_to_native_quil(mapped_program)
            
            print(len(mapped_program))

            #create executable
            #executable = qc.compiler.native_quil_to_executable(mapped_program)
            #qc_init_time = time.time() - start_step

            #qc_time = []
            #qc_start = time.time()
            #bitstrings = qc.run(executable)
            #qc_time.append(time.time() - qc_start)
            #np.save(f, bitstrings)

            #step_time = time.time() - start_step
            #print("Step: %d | total time: %.4f | qc init time: %.4f | qc run time: %.4f" %
            #       (i, step_time, qc_init_time, np.sum(qc_time)))


