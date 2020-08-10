import numpy as np
from arqtic.program import Program, random_bitstring
from arqtic.hamiltonians import Ising_Hamiltonian
from arqtic.arqtic_for_ibm import run_ibm
import qiskit as qk
from qiskit import Aer, IBMQ, execute
import arqtic.QITE as qite
import scipy

N = 3 #number of qubits
beta = 1
shots = 500
dbeta = 0.1 
domain = 3
Jz = -1/np.sqrt(2) #ising interaction strength
mu_x = Jz #transverse magnetic field strength
ising_ham0 = Ising_Hamiltonian(3, Jz, [0.01, 0, 0]) #Hamiltonian at beginning of parameter sweep
psi0 = [1, 0, 0, 0, 0, 0, 0, 0]  

def QITE_prog(ising_ham0, beta, dbeta, domain, psi0):
        psi = psi0
        nbeta = int(beta/dbeta)
        nspins = 1
        #get Pauli basis
        pauli_basis = qite.make_Pauli_basis(domain)
        #get hamiltonian in Pauli basis
        H = qite.get_3QPauliBasis_hamTFIM(Jz, mu_x, nspins, pbc=False)
        #creat array of operators to be exponentiated for QITE
        A_ops = []
        #loop over nbeta steps 
        for ib in range(nbeta):
            print(ib)
            for hterm in H:
                #get the list of qubits this term acts on
                qubits = hterm[0]
                A_ops.append([])
                A_ops[-1].append(qubits)
                #get the array of coeffs for Pauli basis ops that act on these qubits
                h = np.asarray(hterm[1])
                #get coeffs for qite circuit
                x = qite.qite_step3q(psi, pauli_basis, dbeta, h)
                op_coeffs = []
                for i in range(len(x)):
                    if (np.abs(x[i]) > 1e-8):
                        op_coeffs.append(x[i])
                    else: op_coeffs.append(0.0)
                A_ops[-1].append(np.array(op_coeffs))
                op = np.zeros((2**domain,2**domain), dtype=complex)
                for j in range(len(pauli_basis)):
                    op += x[j]*pauli_basis[j]
                print("pauli_basis[23] is: ", pauli_basis[23])
                exp_op = scipy.linalg.expm(1j*op)
                psi = np.dot(exp_op, psi)

                #psi = qite.get_new_psi(psi0, A_ops, pauli_basis, nspins, domain)
                print("new psi is: ", psi)
QITE_prog(ising_ham0, beta, dbeta, domain, psi0)

