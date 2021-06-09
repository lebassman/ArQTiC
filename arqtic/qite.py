import itertools
import numpy as np
from arqtic.program import Program, Gate
from arqtic.exceptions import Error
import scipy
from sklearn.linear_model import Lasso, LinearRegression
import qiskit as qk
from qiskit.aqua.operators.primitive_ops import MatrixOp
from qiskit import Aer, execute
import scipy.linalg as la

#define Pauli matrics
X = np.array([[0.0,1.0],[1.0,0.0]])
Y = np.array([[0,-1.0j],[1.0j,0.0]])
Z = np.array([[1.0,0.0],[0.0,-1.0]])
I = np.eye(2)

gate_matrix_dict = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "I": I,
}

#Pauli I = 0
#Pauli X = 1
#Pauli Y = 2
#Pauli Z = 3
#PauliMultTable1q_ij gives the Pauli (w/o coeff) given by Pauli_i*Pauli_j
#PauliMultCoeffTable1q_ij gives this coefficient
#The 1q denotes that the Paulis along the row/columns are given by 1-qubit Paulis (i.e. I,X,Y,Z)
PauliMultTable1q = np.array([[0, 1, 2, 3],
                               [1, 0, 3, 2],
                               [2, 3, 0, 1],
                               [3, 2, 1, 0]])


PauliMultCoeffTable1q = np.array([[1, 1, 1, 1],
                                   [1, 1, 1j, -1j],
                                   [1, -1j, 1, 1j],
                                   [1, 1j, -1j, 1]])

#When we move to Pauli operators that act on 2qubits (i.e. II,IX,IY,IZ,XI,...,ZZ) we need new tables
PauliMultTable2q=np.zeros((16,16))
for i in range(16):
    for j in range(16):
        PauliMultTable2q[i,j]=int(4*PauliMultTable1q[int(np.floor(i/4)),int(np.floor(j/4))]+PauliMultTable1q[i%4,j%4])

PauliMultCoeffTable2q=np.array(np.zeros((16,16)), dtype=complex)
for i in range(16):
    for j in range(16):
        PauliMultCoeffTable2q[i,j]=PauliMultCoeffTable1q[int(np.floor(i/4)),int(np.floor(j/4))]*PauliMultCoeffTable1q[i%4,j%4]


PauliMultTable3q=np.zeros((64,64))
for i in range(64):
    for j in range(64):
        PauliMultTable3q[i,j]=int(16*PauliMultTable2q[int(np.floor(i/16)),int(np.floor(j/16))]+PauliMultTable2q[i%16,j%16])

PauliMultCoeffTable3q=np.array(np.zeros((64,64)), dtype=complex)
for i in range(64):
    for j in range(64):
        PauliMultCoeffTable3q[i,j]=PauliMultCoeffTable2q[int(np.floor(i/16)),int(np.floor(j/16))]*PauliMultCoeffTable2q[i%16,j%16]


def make_Pauli_basis(domain_size):
    pauli_basis_ops = []
    for s in itertools.product(['I', 'X', 'Y', 'Z'], repeat=domain_size):
        op = [1]
        for d in range(domain_size):
            op = np.kron(op,gate_matrix_dict[s[d]])
        pauli_basis_ops.append(op)
    return pauli_basis_ops


def pauli_basis_names(domain_size):
    pauli_basis_names = []
    for s in itertools.product(['I', 'X', 'Y', 'Z'], repeat=domain_size):
        pauli_basis_names.append(s)
    return pauli_basis_names


def get_PauliBasis_hamTFIM(Jz, mu_x, nspins, domain, pbc=False):
    #check that domain=1 only when nspins=1
    if (domain == 1):
        if (nspins > 1):
            raise Error('Domain can only be set to 1 with nspins=1')  
    #check that domain <= nspins
    if (domain > nspins):
        raise Error('Domain is larger than the number of spins')
    #check that domain < 4
    if (domain > 3):
        raise Error('At this time, domain is limited to 3')
    #define active qubit sets based on domain and nspins
    active_qubit_sets = []
    nsets = nspins-domain+1
    if (pbc ==True): nsets+=1
    for i in range(nsets):
        qset = []
        for j in range(domain):
            qset.append((i+j)%nspins)
        active_qubit_sets.append(qset)
    #build up TFIM hamiltonian in Pauli basis
    H = []
    #create a ham_term for each set of active qubits
    for i in range(nsets):
        hterm = []
        hterm.append(active_qubit_sets[i])
        hm_array = [0]*4**domain
        if (domain == 1):
            #add transverse field term "X" to the one active qubit set
            hm_array[1] = mu_x
        if (domain == 2):
            #add exchange correlation term "ZZ" to all active qubit sets
            hm_array[15] = Jz
            #add transverse field term "XI" to all active qubit sets
            hm_array[4] = mu_x
            #add transverse field term "IX" to last active qubit set if PBC is false
            if (pbc == False):
                if (i == nsets-1):
                    hm_array[1] = mu_x
        if (domain == 3):
            #add exchange correlation term "ZZI" to all active qubit sets
            hm_array[60] = Jz
            #add exchange correlation term "IZZ" to last active qubit set
            if (i == nsets-1):
                hm_array[15] = Jz
            #add transverse field term "XII" to all active qubit sets
            hm_array[16] = mu_x
            #add transverse field term "IXI" to last active qubit set
            if (i == nsets-1):
                hm_array[4] = mu_x
                #add transverse field term "IIX" to last active qubit set if PBC is false
                if (pbc == False):
                    hm_array[1] = mu_x   
        hterm.append(hm_array)
        H.append(hterm)
    return H   

def get_PauliBasis_Heisenberg_Ham(sim_obj, domain, pbc=False):
    nspins = sim_obj.num_spins
    #check that domain=1 only when nspins=1
    if (domain == 1):
        if (nspins > 1):
            raise Error('Domain can only be set to 1 with nspins=1')  
    #check that domain <= nspins
    if (domain > nspins):
        raise Error('Domain is larger than the number of spins')
    #check that domain < 4
    if (domain > 3):
        raise Error('At this time, domain is limited to 3')
    #define active qubit sets based on domain and nspins
    active_qubit_sets = []
    nsets = nspins-domain+1
    if (pbc ==True): nsets+=1
    for i in range(nsets):
        qset = []
        for j in range(domain):
            qset.append((i+j)%nspins)
        active_qubit_sets.append(qset)
    #build up TFIM hamiltonian in Pauli basis
    H = []
    #create a ham_term for each set of active qubits
    for i in range(nsets):
        hterm = []
        hterm.append(active_qubit_sets[i])
        hm_array = [0]*4**domain
        if (domain == 1):
            if len(sim_obj.hx) > 0:
                #add transverse field term "X" to the one active qubit set
                hm_array[1] = sim_obj.hx[0]
            if len(sim_obj.hy) > 0:
                #add transverse field term "Y" to the one active qubit set
                hm_array[2] = sim_obj.hy[0]
            if len(sim_obj.hz) > 0:
                #add transverse field term "Z" to the one active qubit set
                hm_array[3] = sim_obj.hz[0]

        if (domain == 2):
            if len(sim_obj.Jx) > 0:
                #add exchange interaction term "XX" to all active qubit sets
                hm_array[5] = sim_obj.Jx[0]
            if len(sim_obj.Jy) > 0:
                #add exchange interaction term "YY" to all active qubit sets
                hm_array[10] = sim_obj.Jy[0]
            if len(sim_obj.Jz) > 0:
                #add exchange interaction term "ZZ" to all active qubit sets
                hm_array[15] = sim_obj.Jz[0]
            if len(sim_obj.hx) > 0:
                #add transverse field term "XI" to all active qubit sets
                hm_array[4] = sim_obj.hx[0]
                #add transverse field term "IX" to last active qubit set if PBC is false
                if (pbc == False):
                    if (i == nsets-1):
                        hm_array[1] = sim_obj.hx[0]
            if len(sim_obj.hy) > 0:
                #add transverse field term "YI" to all active qubit sets
                hm_array[8] = sim_obj.hy[0]
                #add transverse field term "IY" to last active qubit set if PBC is false
                if (pbc == False):
                    if (i == nsets-1):
                        hm_array[2] = sim_obj.hy[0]
            if len(sim_obj.hz) > 0:
                #add transverse field term "ZI" to all active qubit sets
                hm_array[12] = sim_obj.hz[0]
                #add transverse field term "IZ" to last active qubit set if PBC is false
                if (pbc == False):
                    if (i == nsets-1):
                        hm_array[3] = sim_obj.hz[0]
                        
        if (domain == 3):
            if len(sim_obj.Jx) > 0:
                #add exchange interaction term "XXI" to all active qubit sets
                hm_array[20] = sim_obj.Jx[0]
                #add exchange interaction term "IXX" to last active qubit set
                if (i == nsets-1):
                    hm_array[5] = sim_obj.Jx[0]
            if len(sim_obj.Jy) > 0:
                #add exchange interaction term "YYI" to all active qubit sets
                hm_array[40] = sim_obj.Jy[0]
                #add exchange interaction term "IYY" to last active qubit set
                if (i == nsets-1):
                    hm_array[10] = sim_obj.Jy[0]
            if len(sim_obj.Jz) > 0:
                #add exchange interaction term "ZZI" to all active qubit sets
                hm_array[60] = sim_obj.Jz[0]
                #add exchange interaction term "IZZ" to last active qubit set
                if (i == nsets-1):
                    hm_array[15] = sim_obj.Jz[0]
            if len(sim_obj.hx) > 0:     
                #add transverse field term "XII" to all active qubit sets
                hm_array[16] = sim_obj.hx[0]
                #add transverse field term "IXI" to last active qubit set
                if (i == nsets-1):
                    hm_array[4] = sim_obj.hx[0]
                    #add transverse field term "IIX" to last active qubit set if PBC is false
                    if (pbc == False):
                        hm_array[1] = sim_obj.hx[0]
            if len(sim_obj.hy) > 0:     
                #add transverse field term "YII" to all active qubit sets
                hm_array[32] = sim_obj.hy[0]
                #add transverse field term "IYI" to last active qubit set
                if (i == nsets-1):
                    hm_array[8] = sim_obj.hy[0]
                    #add transverse field term "IIY" to last active qubit set if PBC is false
                    if (pbc == False):
                        hm_array[2] = sim_obj.hy[0]
            if len(sim_obj.hz) > 0:      
                #add transverse field term "ZII" to all active qubit sets
                hm_array[48] = sim_obj.hz[0]
                #add transverse field term "IZI" to last active qubit set
                if (i == nsets-1):
                    hm_array[12] = sim_obj.hz[0]
                    #add transverse field term "IIZ" to last active qubit set if PBC is false
                    if (pbc == False):
                        hm_array[3] = sim_obj.hz[0]
        hterm.append(hm_array)
        H.append(hterm)
    return H   


def get_exepctation_values_th(psi, Pauli_basis, active_qubits, nspins):
    exp_values = []
    for i in range(len(Pauli_basis)):
        #enable pauli_basis operator to act on entire qubit system
        full_op = [1]
        pauli_op_not_applied = True
        for k in range(nspins):
            if (k in active_qubits):
                if(pauli_op_not_applied): 
                    full_op = np.kron(full_op, Pauli_basis[i])
                    pauli_op_not_applied = False
            else:
                full_op = np.kron(full_op,np.eye(2))
        #get expectation value of full pauli operator in state psi
        exp_values.append(np.real(np.dot(np.transpose(np.conj(psi)),np.dot(full_op,psi))))
    return exp_values


def get_energy_from_exps(exp_vals, ham):
    energy = 0
    for i in range(len(exp_vals)): 
            energy += exp_vals[i]*ham[i]
    return energy

def compute_norm(expectation_values, dbeta, ham, domain):
    norm = 0
    Pm_coeffs = -dbeta*ham
    Pm_coeffs[0] += 1
    for i, j in itertools.product(range(len(ham)), repeat=2):
        if (domain == 1):
            norm += np.conj(Pm_coeffs[i])*Pm_coeffs[j]*PauliMultCoeffTable1q[i,j]*expectation_values[int(PauliMultTable1q[i,j])]
        if (domain == 2):
            norm += np.conj(Pm_coeffs[i])*Pm_coeffs[j]*PauliMultCoeffTable2q[i,j]*expectation_values[int(PauliMultTable2q[i,j])]
        if (domain == 3):
            norm += np.conj(Pm_coeffs[i])*Pm_coeffs[j]*PauliMultCoeffTable3q[i,j]*expectation_values[int(PauliMultTable3q[i,j])]
    return np.sqrt(norm)    


def compute_Smatrix(expectation_values, ham, domain):
    dim = len(expectation_values)
    S=np.zeros((dim,dim))
    for i, j in itertools.product(range(len(ham)), repeat=2):
        if (domain == 1):
            S[i,j] = 2*np.real(PauliMultCoeffTable1q[i,j]*expectation_values[int(PauliMultTable1q[i,j])])
        if (domain == 2):
            S[i,j] = 2*np.real(PauliMultCoeffTable2q[i,j]*expectation_values[int(PauliMultTable2q[i,j])])
        if (domain == 3):
            S[i,j] = 2*np.real(PauliMultCoeffTable3q[i,j]*expectation_values[int(PauliMultTable3q[i,j])])
    return S


def compute_bvec(expectation_values, dbeta, ham, norm, domain):
    dim = len(expectation_values)
    b = np.zeros(dim)
    Pm = -dbeta*ham
    Pm[0] += 1
    for i, j in itertools.product(range(len(ham)), repeat=2):
        if (domain == 1):
            b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable1q[j,i]*expectation_values[int(PauliMultTable1q[i,j])])
        if (domain == 2):
            b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable2q[j,i]*expectation_values[int(PauliMultTable2q[i,j])])
        if (domain == 3):
            b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable3q[j,i]*expectation_values[int(PauliMultTable3q[i,j])])
    return b



def qite_step(psi, pauli_basis, active_qubits, nspins, dbeta, ham, domain,regularizer):
    #get expectation values of Pauli basis operators for state psi
    exp_values = get_exepctation_values_th(psi, pauli_basis, active_qubits, nspins)
    #get energy from exp_values
    energy = get_energy_from_exps(exp_values, ham)
    #compute S matrix
    S_mat = compute_Smatrix(exp_values, ham, domain)
    #compute norm of sum of Pauli basis ops on psi
    norm = compute_norm(exp_values, dbeta, ham, domain)
    #compute b-vector 
    b_vec = compute_bvec(exp_values, dbeta, ham, norm, domain)
    #solve linear equation for x
    #dalpha = np.eye(len(pauli_basis))*regularizer
    #x = np.linalg.lstsq(S_mat + dalpha, -b_vec, rcond=-1)[0]
    #x = np.linalg.lstsq(S_mat,-b_vec,rcond=-1)[0]
    #clf = Lasso(alpha=regularizer)
    reg = LinearRegression()
    reg.fit(S_mat, b_vec)
    x = reg.coef_ 
    return x, energy



def get_new_psi(psi0, A_ops, pauli_basis, nspins, domain):
    psi = psi0
    for i in range(len(A_ops)):
        active_qubits = A_ops[i][0]
        op = np.zeros((2**domain,2**domain), dtype=complex)
        for j in range(len(pauli_basis)):
            op += A_ops[i][1][j]*pauli_basis[j]
        #exponentiate op 
        exp_op = scipy.linalg.expm(1j*op)
        #exp_op just acts on active qubits so convert to op that acts on whole system
        exp_op_full = [1]
        exp_op_not_applied = True
        for k in range(nspins):
            if (k in active_qubits):
                if(exp_op_not_applied): 
                    exp_op_full = np.kron(exp_op_full, exp_op)
                    exp_op_not_applied = False
            else:
                exp_op_full = np.kron(exp_op_full,np.eye(2))
        psi = np.dot(exp_op_full, psi)
        psi = psi/np.linalg.norm(psi)
    return psi


def get_state_from_string(string):
    psi = [1]
    #rev_str = str(string)[::-1]
    for q in string:
        if (q == '0'):
            psi = np.kron(psi,np.array([1,0]))
        elif (q == '1'):
            psi = np.kron(psi,np.array([0,1]))
        else: 
            print("bad value for initial psi")
    return psi


def Aop_to_Terms(A, domain):
    nqubits = len(A[0])
    nops = len(A[1])
    names = pauli_basis_names(domain)
    terms = []
    for i in range(nops):
        coeff = A[1][i]
        if (abs(coeff) > 1e-12):
            paulis = []
            for j in range(domain):
                if (names[i][j] != "I"):
                    paulis.append(prog.Pauli(names[i][j],A[0][j]))
            term = prog.Term(paulis, coeff)
            if (len(paulis) > 0):
                terms.append(term)
    return terms

def Aop_to_matrix(A, domain):
    unitary_mat = []
    qubits = A[0]
    unitary_mat.append(qubits)
    coeffs = A[1]
    names = pauli_basis_names(domain)
    total_mat = np.zeros((2**domain, 2**domain), dtype=complex)
    for i in range(len(coeffs)):
        coeff = coeffs[i]
        pauli_mat = [1]
        for j in range(domain):
            pauli = gate_matrix_dict[names[i][j]]
            pauli_mat = np.kron(pauli_mat, pauli)
        pauli_mat *= coeff
        total_mat += pauli_mat
            
    unitary_mat.append(total_mat)
    return unitary_mat



def make_QITE_program(sim_obj, regularizer=0.1):
    beta = sim_obj.beta
    dbeta = sim_obj.delta_beta
    domain = sim_obj.domain
    backend = sim_obj.backend
    nbeta = int(beta/dbeta)
    psi = get_state_from_string(sim_obj.initial_spins)
    nspins = sim_obj.num_spins
    #get Pauli basis
    pauli_basis = make_Pauli_basis(domain)
    #get hamiltonian in Pauli basis
    H = get_PauliBasis_Heisenberg_Ham(sim_obj, domain, pbc=False)
    #creat array of operators to be exponentiated for QITE
    A_ops = []
    energies = []
    qite_prog = Program(nspins)
    #add initial state preparation to program
    spin_idx = 0
    for spin in sim_obj.initial_spins:
        if (spin == '1'):
            qite_prog.add_gate(Gate([spin_idx], name='X'))
        spin_idx += 1
    #loop over nbeta steps 
    for ib in range(nbeta):
        total_eng = 0
        for hterm in H:
            #get the list of qubits this term acts on
            active_qubits = hterm[0]
            A_ops.append([])
            A_ops[-1].append(active_qubits)
            #get the array of coeffs for Pauli basis ops that act on these qubits
            ham = np.asarray(hterm[1])
            #get coeffs for qite circuit
            x, energy = qite_step(psi, pauli_basis, active_qubits, nspins, dbeta, ham, domain,regularizer)
            total_eng += energy
            op_coeffs = []
            for i in range(len(x)):
                if (np.abs(x[i]) > 1e-12):
                    op_coeffs.append(x[i])
                else: op_coeffs.append(0.0)
            x = op_coeffs
            A_ops[-1].append(x)
            Xmat=np.complex128(np.zeros([2**domain,2**domain]))
            for p in range(4**domain):
                Xmat+=np.complex128(x[p])*pauli_basis[p]
            exp_op = scipy.linalg.expm(-1j*Xmat)
            exp_op_full = [1]
            exp_op_not_applied = True
            for k in range(nspins):
                if (k in active_qubits):
                    if(exp_op_not_applied): 
                        exp_op_full = np.kron(exp_op_full, exp_op)
                        exp_op_not_applied = False
                else:
                    exp_op_full = np.kron(exp_op_full,np.eye(2))

            psi = np.dot(exp_op_full, psi)
            #psi = psi/np.linalg.norm(psi)
            #add unitary to qite program
            #qite_prog.add_gate(Gate(active_qubits, unitary=exp_op_full))
            qite_prog.add_gate(Gate(active_qubits, unitary=exp_op))
        energies.append(total_eng)
    final_eng = 0
    for hterm in H:
        active_qubits = hterm[0]
        ham = np.asarray(hterm[1])    
        exp_vals = get_exepctation_values_th(psi, pauli_basis, active_qubits, nspins)
        final_eng += get_energy_from_exps(exp_vals, ham)
    energies.append(final_eng)

    return qite_prog, energies
