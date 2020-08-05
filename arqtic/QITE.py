import itertools
import numpy as np

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


def make_Pauli_basis(domain_size):
    pauli_basis_ops = []
    for s in itertools.product(['I', 'X', 'Y', 'Z'], repeat=domain_size):
        op = [1]
        for d in range(domain_size):
            op = np.kron(op,gate_matrix_dict[s[d]])
        pauli_basis_ops.append(op)
    return pauli_basis_ops

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
        hterm.append([])
        hterm[-1].append([0,nspins-1])
        hm_array = [0]*16
        hm_array[15] = -Jz 
        hterm[-1].append(hm_array)
        H.append(hterm)
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


def compute_Amatrix(expectation_values):
    dim = len(expectation_values)
    A=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            A[i,j] = 2*np.real(PauliMultCoeffTable2q[i,j]*expectation_values[int(PauliMultTable2q[i,j])])
    return A


def compute_bvec(expectation_values, dbeta, h, norm):
    dim = len(expectation_values)
    b = np.zeros(dim) 
    Pm = -dbeta*h
    Pm[0] += 1
    for i in range(dim):
        for j in range(dim):
            b[i]+=2*np.imag((np.conj(Pm[j])/norm)*PauliMultCoeffTable2q[j,i]*expectation_values[int(PauliMultTable2q[i,j])])
    return b


psi = [1,0,0,0]
pauli_basis = make_Pauli_basis(2)
exp_values = get_exepctation_values_th(psi, pauli_basis)
H = get_2QPauliBasis_hamTFIM(1.0, 2.0, 4, pbc=False) 
dbeta = 0.5
for hterm in H:
    h = np.asarray(hterm[1])
    print(h)
    norm = compute_norm(exp_values, 0.5,h)
    print(norm)
    print(compute_Amatrix(exp_values))
    print(compute_bvec(exp_values, dbeta, h, norm))


