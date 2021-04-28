from arqtic.program import Program
from arqtic.arqtic_for_ibm import ibm_circ_to_program
import qsearch
from qsearch.gates import *
from qsearch.unitaries import *
from qsearch.assemblers import *
from qsearch import multistart_solvers, utils, options, leap_compiler, post_processing, assemblers
from qsearch.defaults import standard_defaults, standard_smart_defaults


def make_matchgate():
    u3 = U3Gate()
    cnot = CNOTGate()
    I = IdentityGate()

    u3_layer = KroneckerGate(u3,u3)
    matchgate = ProductGate(u3_layer, cnot, u3_layer, cnot, u3_layer)
    return matchgate
    
def make_layertype1(N):
    I = IdentityGate()
    for i in range(int(N/2)):
        if (i==0):
            layer = make_matchgate()
        else:
            layer = KroneckerGate(layer,make_matchgate())
    if (N%2 != 0): 
        layer = KroneckerGate(layer, I)
    return layer
        
def make_layertype2(N):
    I = IdentityGate()
    layer = I
    #N even
    if (N%2==0):
        for _ in range(int(N/2)-1):
            layer = KroneckerGate(layer,make_matchgate())
        layer = KroneckerGate(layer,I)
    #N odd
    else:
        for _ in range(int(N/2)):
            layer = KroneckerGate(layer,make_matchgate())
    return layer

def make_MGC(N):
    """
    Make matchgate circuit for N qubits.
    Args:
        N (int): The number of spins.
    Returns
        circuit (ProductGate): Circuit of QSearch gates.
    """
    for l in range(N):
        #add layer_type1
        if (l%2 == 0):
            if(l==0):
                circuit = make_layertype2(N)
            else:
                circuit = ProductGate(circuit, make_layertype2(N))
        #add layer_type2
        else:
            circuit = ProductGate(circuit, make_layertype1(N))
    return circuit


def get_constant_depth_program(program, N, multistarts=24, optimizer_repeats=5):
    target_unitary = program.get_U()
    #get constant-depth circuit structure for N qubits
    circ_struct = make_MGC(N)
    #optimize parameters of circuit
    # use the multistart solver, may want to increase the number of starts for more qubits, but that will also be slower
    solv = multistart_solvers.MultiStart_Solver(multistarts)
    # set up some options
    opts = qsearch.Options()
    opts.target = target_unitary
    #opts.gateset = gateset
    opts.set_defaults(**standard_defaults)
    opts.set_smart_defaults(**standard_smart_defaults)
    # optimize the circuit structure (circ_struct) for target U
    # returns the calculated matrix and the vector of parameters
    dist = 1
    # run a few times to make sure we find the correct solution
    for _ in range(optimizer_repeats):
        mat, vec = solv.solve_for_unitary(circ_struct, opts)
        dist_new = utils.matrix_distance_squared(mat, target_unitary)
        print(dist_new)
        if dist_new < dist:
            dist = dist_new
        if dist < 1e-10:
            break

    print(f'Got distance {dist}')
    #get final program
    result_dict = {}
    result_dict["structure"] = circ_struct
    result_dict["parameters"] = vec
    
    opts.assemblydict=assemblydict_ibmopenqasm
    out = opts.assembler.assemble(result_dict, opts)
    prog = ibm_circ_to_program(out)
    return prog
   
    
def get_constant_depth_ibm_circuit(program, N, multistarts=24, optimizer_repeats=5):
    target_unitary = program.get_U()
    #get constant-depth circuit structure for N qubits
    circ_struct = make_MGC(N)
    #optimize parameters of circuit
    # use the multistart solver, may want to increase the number of starts for more qubits, but that will also be slower
    solv = multistart_solvers.MultiStart_Solver(multistarts)
    # set up some options
    opts = qsearch.Options()
    opts.target = target_unitary
    #opts.gateset = gateset
    opts.set_defaults(**standard_defaults)
    opts.set_smart_defaults(**standard_smart_defaults)
    # optimize the circuit structure (circ_struct) for target U
    # returns the calculated matrix and the vector of parameters
    dist = 1
    # run a few times to make sure we find the correct solution
    for _ in range(optimizer_repeats):
        mat, vec = solv.solve_for_unitary(circ_struct, opts)
        dist_new = utils.matrix_distance_squared(mat, target_unitary)
        print(dist_new)
        if dist_new < dist:
            dist = dist_new
        if dist < 1e-10:
            break

    print(f'Got distance {dist}')
    #get final program
    result_dict = {}
    result_dict["structure"] = circ_struct
    result_dict["parameters"] = vec
    
    opts.assemblydict=assemblydict_ibmopenqasm
    out = opts.assembler.assemble(result_dict, opts)
    ibm_circ = qk.QuantumCircuit.from_qasm_str(out)
    return ibm_circ