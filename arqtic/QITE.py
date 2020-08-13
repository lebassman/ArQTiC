import itertools
import numpy as np
import arqtic.program as prog
import scipy
from sklearn.linear_model import Lasso


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
for i1 in range(64):
    for i2 in range(64):
        PauliMultTable3q[i1,i2]=int(16*PauliMultTable2q[int(np.floor(i1/16)),int(np.floor(i2/16))]+PauliMultTable2q[i1%16,i2%16])

PauliMultCoeffTable3q=np.array(np.zeros((64,64)), dtype=complex)
for i1 in range(64):
    for i2 in range(64):
        PauliMultCoeffTable3q[i1,i2]=PauliMultCoeffTable2q[int(np.floor(i1/16)),int(np.floor(i2/16))]*PauliMultCoeffTable2q[i1%16,i2%16]

  


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


def get_2QPauliBasis_hamTFIM(Jz, mu_x, nspins, pbc=False):
    H = []
    for i in range(nspins-1):
        hterm = []
        hterm.append([i,i+1]) #define active qubits for term
        hm_array = [0]*16
        hm_array[4] = -mu_x
        hm_array[15] = -Jz
        if (i == nspins-2): hm_array[1] = -mu_x
        hterm.append(hm_array)
        H.append(hterm)
    if(pbc):
        hterm = []
        hterm.append([0,nspins-1])
        hm_array = [0]*16
        hm_array[15] = -Jz 
        hterm[-1].append(hm_array)
        H.append(hterm)
    return H


def get_3QPauliBasis_hamTFIM(Jz, mu_x, nspins, pbc=False):
    H = []
    hterm = []
    hterm.append([0,1,2]) #define active qubits for term
    hm_array = [0]*64
    hm_array[1] = -mu_x
    hm_array[4] = -mu_x
    hm_array[16] = -mu_x
    hm_array[15] = -Jz
    hm_array[60] = -Jz
    hterm.append(hm_array)
    H.append(hterm)
    #if(pbc):
    #    hterm = []
    #    hterm.append([0,nspins-1])
    #    hm_array = [0]*16
    #    hm_array[15] = -Jz
    #    hterm[-1].append(hm_array)
    #    H.append(hterm)
    return H


def get_exepctation_values_th(psi, Pauli_basis):
    exp_values = []
    for i in range(len(Pauli_basis)):
        exp_values.append(np.dot(psi,np.dot(Pauli_basis[i],psi)))
    return exp_values


def compute_norm(expectation_values, dbeta, h):
    norm = 0
    Pm_coeffs = -dbeta*h
    Pm_coeffs[0] += 1
    for i in range(len(h)):
        for j in range(len(h)):
            norm += np.conj(Pm_coeffs[i])*Pm_coeffs[j]*PauliMultCoeffTable2q[i,j]*expectation_values[int(PauliMultTable2q[i,j])]
    return np.sqrt(norm)    


def compute_norm3q(expectation_values, dbeta, h):
    norm = 0
    Pm_coeffs = -dbeta*h
    Pm_coeffs[0] += 1
    for i, j in itertools.product(range(len(h)), repeat=2):
        norm += np.conj(Pm_coeffs[i])*Pm_coeffs[j]*PauliMultCoeffTable3q[i,j]*expectation_values[int(PauliMultTable3q[i,j])]
    return np.sqrt(norm)

def compute_Smatrix(expectation_values):
    dim = len(expectation_values)
    S=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            S[i,j] = 2*np.real(PauliMultCoeffTable2q[i,j]*expectation_values[int(PauliMultTable2q[i,j])])
    return S


def compute_Smatrix3q(expectation_values, h):
    dim = len(expectation_values)
    S=np.zeros((dim,dim))
    for i, j in itertools.product(range(len(h)), repeat=2):
        S[i,j] = 2*np.real(PauliMultCoeffTable3q[i,j]*expectation_values[int(PauliMultTable3q[i,j])])
    return S

def compute_bvec(expectation_values, dbeta, h, norm):
    dim = len(expectation_values)
    b = np.zeros(dim) 
    Pm = -dbeta*h
    Pm[0] += 1
    for i in range(dim):
        for j in range(dim):
            b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable2q[j,i]*expectation_values[int(PauliMultTable2q[i,j])])
    return b

def compute_bvec3q(expectation_values, dbeta, h, norm):
    dim = len(expectation_values)
    b = np.zeros(dim)
    Pm = -dbeta*h
    Pm[0] += 1
    for i, j in itertools.product(range(len(h)), repeat=2):
        b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable3q[j,i]*expectation_values[int(PauliMultTable3q[i,j])])
    return b


def qite_step(psi, pauli_basis, dbeta, h):
    #get expectation values of Pauli basis operators for state psi
    exp_values = get_exepctation_values_th(psi, pauli_basis)
    #print("exp_values is: ", exp_values)
    #compute S matrix
    S_mat = compute_Smatrix(exp_values)
    #compute norm of sum of Pauli basis ops on psi
    norm = compute_norm(exp_values, dbeta, h)
    #print("norm is: ", norm)
    #print("h is: ", h)
    #compute b-vector 
    b_vec = compute_bvec(exp_values, dbeta, h, norm)
    #solve linear equation for x
    #dalpha = np.eye(len(pauli_basis))*regularizer
    #x = np.linalg.lstsq(S_mat,-b_vec,rcond=-1)[0]
    clf = Lasso(alpha=0.001)
    clf.fit(S_mat, b_vec)
    x = clf.coef_ 
    #print("Smat is: ", S_mat)
    #print("bvec is: ", b_vec)
    #print("x is: ", x)
    return x


def qite_step3q(psi, pauli_basis, dbeta, h):
    #get expectation values of Pauli basis operators for state psi
    exp_values = get_exepctation_values_th(psi, pauli_basis)
    print("exp_values is: ", exp_values)
    #compute S matrix
    S_mat = compute_Smatrix3q(exp_values,h)
    #compute norm of sum of Pauli basis ops on psi
    norm = compute_norm3q(exp_values, dbeta, h)
    print("norm is: ", norm)
    print("ham is: ", h)
    #compute b-vector 
    b_vec = compute_bvec3q(exp_values, dbeta, h, norm)
    #solve linear equation for x
    #dalpha = np.eye(len(pauli_basis))*regularizer
    #x = np.linalg.lstsq(S_mat,-b_vec,rcond=-1)[0]
    clf = Lasso(alpha=0.001)
    clf.fit(S_mat, b_vec)
    x = clf.coef_
    print("Smat is: ", S_mat)
    print("bvec is: ", b_vec)
    print("x is: ", x)
    return x


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
    return psi


def get_state_from_string(string):
    psi = [1]
    for q in range(len(string)):
        if (string[q] == 0):
            psi = np.kron(psi,np.array([1,0]))
        else:
            psi = np.kron(psi,np.array([0,1]))
    return psi


def Aop_to_Terms(A, domain):
    nqubits = len(A[0])
    nops = len(A[1])
    names = pauli_basis_names(domain)
    terms = []
    for i in range(nops):
        if (np.abs(A[1][i]) > 1e-8):
            coeff = A[1][i]
            paulis = []
            for j in range(domain):
                if (names[i][j] != "I"):
                    paulis.append(prog.Pauli(names[i][j],A[0][j]))
            term = prog.Term(paulis, coeff)
            if (len(paulis) > 0):
                terms.append(term)
    return terms


#psi0 = [1,1]
#psi = get_state_from_string(psi0)
#pauli_basis = make_Pauli_basis(2)
#names = pauli_basis_names(2)
#exp_values = get_exepctation_values_th(psi, pauli_basis)
#H = get_2QPauliBasis_hamTFIM(1.0, 2.0, 4, pbc=False) 
#dbeta = 0.5
#for hterm in H:
#    h = np.asarray(hterm[1])
#    print(h)
#    norm = compute_norm(exp_values, 0.5,h)
#    print(norm)
#    print(compute_Smatrix(exp_values))
#    print(compute_bvec(exp_values, dbeta, h, norm))


