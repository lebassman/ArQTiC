import numpy as np
import random
from .hamiltonians import Ising_Hamiltonian	

#define gate  matrices
X = np.array([[0.0,1.0],[1.0,0.0]])
Y = np.array([[0,-1.0j],[1.0j,0.0]])
Z = np.array([[1.0,0.0],[0.0,-1.0]])
I = np.eye(2)
H = np.array([[1.0,1.0],[1.0,-1.0]])*(1/np.sqrt(2.0))
CNOT = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
CZ = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,-1.0]])

def RX(theta):
    return np.array([[np.cos(theta/2.0) , -1.0j*np.sin(theta/2.0)], [-1.0j*np.sin(theta/2.0), np.cos(theta/2.0)]])

def RY(theta):
    return np.array([[np.cos(theta/2.0) , -np.sin(theta/2.0)], [np.sin(theta/2.0), np.cos(theta/2.0)]])

def RZ(theta):
    return np.array([[np.exp(-1.0j*theta/2.0), 0], [0, np.exp(1.0j*theta/2.0)]])

gate_matrix_dict = {
    "X": X, 
    "Y": Y,
    "Z": Z, 
    "I": I, 
    "H": H, 
    "CNOT": CNOT, 
    "CZ": CZ
}

class Gate: 
    def __init__(self, name, qubits, angles=[]):
        self.name = name
        self. angles = angles
        self.qubits = qubits

    def matrix(self):
        if self.name in gate_matrix_dict:
            return gate_matrix_dict[self.name]
        elif (self.name == 'RX'):
            return RX(self.angles[0])
        elif (self.name == 'RY'):
            return RY(self.angles[0])
        elif (self.name == 'RZ'):
            return RZ(self.angles[0])
        else:
            print("Error: ", self.name, " is not a known gate name!")
            exit()

    def gate_from_Pauli(self, pauli):
        self.name = pauli.name
        self.qubits = pauli.qubit
        self.angles = []

    def print_gate(self):
        if (self.angles != []):
            print(self.name,"(",self.angles[0],")",self.qubits) 
        else: 
            print(self.name,self.qubits)  

class Pauli: 
    def __init__(self, name, qubit):
        self.name = name
        self.qubit = qubit

class Term:
    def __init__(self, paulis, coeff):
        self.paulis = paulis
        self.coeff = coeff

class Program:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.gates = [] 

    def add_instr(self, gate_list):
        for gate in gate_list:
            self.gates.append(gate)
            
    def append_program(self, program):
        self.nqubits = max(self.nqubits, program.nqubits)
        for gate in program.gates:
            self.gates.append(gate)
        
    def get_U(self):
        dim = 2**self.nqubits
        matU = np.eye(dim)
        for gate in self.gates:
            #make sure gate matrix has dimension of system
            if(self.nqubits > 1):
                kron_list = [I] * self.nqubits
                kron_list[gate.qubits[0]] = gate.matrix()
                num_prods = self.nqubits-1
                if (len(gate.qubits) == 2):
                    num_prods = self.nqubits-2
                    if (self.nqubits > 2):
                        #for a 2-qubit gate, the last identity should be removed from kron_list
                        kron_list.pop()
                    else: mat = gate.matrix()
                for q in range(num_prods):
                    if (q==0):
                        mat = np.kron(kron_list[0], kron_list[1])
                    else:
                        mat = np.kron(mat, kron_list[q+1])
            else: mat = gate_dict[gate.matrix]
            matU = np.matmul(matU,mat)
        return matU     

    def print_list(self):
        for gate in self.gates:
            gate.print_gate()
            
    def make_ps_prog(self, bitstring):
        for i in range(len(bitstring)):
            if (bitstring[i] == 1):
                self.gates.append(Gate("X", [i]))
                
    def make_rand_measurement_prog(self):
        for q in range(self.nqubits):
            self.gates.append(Gate(random.choice(["I", "X", "Y", "Z"]), [q]))
            
                
    def make_hamEvol_prog(self, time_step, dtau, dt, lambda_protocol, ising_ham):
        trotter_steps = int(dtau/dt) #number of Trotter-steps per time-step
        theta_z = 2.0*ising_ham.exchange_coeff*dt
        for step in range(time_step):
            #apply external magnetic field term in x-dir
            theta_x = 2*ising_ham.ext_mag_vec[0]*lambda_protocol[step]*dt
            for m in range(trotter_steps):
                for q in range(self.nqubits):
                    self.gates.append(Gate("H", [q]))
                    self.gates.append(Gate("RZ", [q], [theta_x])) 
                    self.gates.append(Gate("H", [q])) 
                    #apply coupling term
                for q in range(self.nqubits):
                    self.gates.append(Gate("CNOT", [q,q+1]))
                    self.gates.append(Gate("RZ", [q], [theta_z]))
                    self.gates.append(Gate("CNOT", [q,q+1]))
        

    def get_Qcompile_input(self, filename='QCompile_input.txt'):
        matU = self.get_U()
        dim = 2**self.nqubits
        f = open(filename, "w")
        f.write(str(dim))
        f.write("\n")
        for i in range(dim):
            for j in range(dim):
                f.write(str(np.real(matU[i][j])))
                f.write(" ")
                f.write(str(np.imag(matU[i][j])))
                f.write("\n")

def random_bitstring(nbits):
    bitstring = []
    for i in range(nbits):
        bitstring.append(random.choice([0,1]))
    return bitstring
    

    
    
    


#J = 0.01183898
#my_Ising = Ising_Hamiltonian(4, J, [J,0.0,0.0])
#my_Ising.print_pretty()
#program = my_Ising.get_Trotter_program(3, 9)
#program.print_list()
#program.get_Qcompile_input()




#p = Program(2)
#gate1 = Gate('RZ',[0], angles=[0.4])
#gate2 = Gate('CNOT', [0,1])
#gate3 = Gate('CNOT', [4,5])
#gate_list = [gate1, gate2]
#p.add_instr(gate_list)
#print(p.get_U())