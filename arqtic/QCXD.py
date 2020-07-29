#Connor Powers
#Note: This code was initially based on Lindsay Bassman's Ising model code.

#General usage flow:
#Create input file
#from command line: 
#   from H_simulator import *
#   arbitrary_name=Heisenberg()   or Heisenberg(specific_file_name)
#   arbitrary_name.connect_account()    (you only have to do this once at the beginning of a session)
#   arbitrary_name.run()


#import necessary libraries
import sys
import numpy as np
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout

from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
import matplotlib.pyplot as plt
import os
from qiskit.circuit import quantumcircuit
from qiskit.circuit import Instruction

current=os.getcwd()
newdir="Data"
path = os.path.join(current, newdir) 
if not os.path.isdir(path):
    os.makedirs(path)


filename="logfile"
completename = os.path.join(path,filename+".txt")




#Set up the Hamiltonian.py-based tools#################################################################################################################################
#define gate  matrices
X = np.array([[0.0,1.0],[1.0,0.0]])
Y = np.array([[0,-1.0j],[1.0j,0.0]])
Z = np.array([[1.0,0.0],[0.0,-1.0]])
I = np.eye(2)
H = np.array([[1.0,1.0],[1.0,-1.0]])*(1/np.sqrt(2.0))
CNOT = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]])
CZ = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,-1.0]])

gate_dict = {
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
        if self.name in gate_dict:
            return gate_dict[self.name]
        elif (self.name == 'RZ'):
            return np.array([[np.cos(self.angles[0]/2)-1.0j*np.sin(self.angles[0]/2), 0],[0, np.cos(self.angles[0]/2)+1.0j*np.sin(self.angles[0]/2)]])
        elif (self.name == 'RX'):
            return np.array([[np.cos(self.angles[0]/2), -1.0j*np.sin(self.angles[0]/2)],[-1.0j*np.sin(self.angles[0]/2), np.cos(self.angles[0]/2)]])
        elif (self.name == 'RY'):
            return np.array([[np.cos(self.angles[0]/2), -np.sin(self.angles[0]/2)],[np.sin(self.angles[0]/2), np.cos(self.angles[0]/2)]])
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

class Hamiltonian:
    def __init__(self, nqubits, terms):
        self.nqubits = nqubits
        self.terms = terms

    def add_term(self, term):
        self.term.append(term)

    def matrix(self):
        dim = 2**self.nqubits
        ham_mat = np.zeros((dim,dim))
        for term in self.terms:
            kron_list = ['I'] * self.nqubits
            for pauli in term.paulis:
               kron_list[pauli.qubit] = pauli.name
            for q in range(self.nqubits-1):
                if (q==0):
                    mat = np.kron(gate_dict[kron_list[0]], gate_dict[kron_list[1]])
                else:          
                    mat = np.kron(mat, gate_dict[kron_list[q+1]])
            mat = term.coeff * mat 
            ham_mat = ham_mat + mat
        return ham_mat
    
    def show(self):
        dim = 2**self.nqubits
        ham_mat = self.matrix()
        for i in range(dim):
            for j in range(dim):
                print(ham_mat[i][j])

class Ising_Hamiltonian:
    def __init__(self, nqubits, exchange_coeff, ext_mag_vec, pbc=False):
        self.nqubits = nqubits
        self.exchange_coeff = exchange_coeff
        self.ext_mag_vec = ext_mag_vec
        self.pbc = pbc

    def matrix(self):
        ham_terms = []
        x_field = self.ext_mag_vec[0]
        y_field = self.ext_mag_vec[1]
        z_field = self.ext_mag_vec[2]
        #add Ising exchange interaction terms to Hamiltonian
        for q in range(self.nqubits-1):
            pauli1 = Pauli('Z',q)
            pauli2 = Pauli('Z',q+1)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #in case of pbc=true add term n->0
        if(self.pbc): 
            pauli1 = Pauli('Z',self.nqubits-1)
            pauli2 = Pauli('Z',0)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #add external magnetic field terms to Hamiltonian
        for q in range(self.nqubits):
            if (x_field != 0.0):
                pauli = Pauli('X', q)
                term = Term([pauli], -1.0*x_field)
                ham_terms.append(term)
            if (y_field != 0.0):
                pauli = Pauli('Y', q)
                term = Term([pauli], -1.0*y_field)
                ham_terms.append(term)
            if (z_field != 0.0):
                pauli = Pauli('Z', q)
                term = Term([pauli], -1.0*z_field)
                ham_terms.append(term)
        ham_mat = Hamiltonian(self.nqubits, ham_terms)
        return ham_mat.matrix()



    def get_Trotter_program(self, delta_t, total_time): #right now only works for x-dir external magnetic field and pbc=False
        H_BAR = 0.658212 #eV*fs 
        p = Program(self.nqubits)
        timesteps = int(total_time/delta_t)
        for t in range(0,timesteps):
            instr_set1 = []
            instr_set2 = []
            for q in range(0, self.nqubits):
                instr_set1.append(Gate('H', [q]))
                instr_set1.append(Gate('RZ', [q], angles=[(-2.0*self.ext_mag_vec[0]*delta_t/H_BAR)]))
                instr_set1.append(Gate('H',[q]))
            for q in range(0, self.nqubits-1):
                instr_set2.append(Gate('CNOT',[q, q+1]))
                instr_set2.append(Gate('RZ', [q+1], angles=[-2.0*self.exchange_coeff*delta_t/H_BAR]))
                instr_set2.append(Gate('CNOT', [q, q+1]))
            p.add_instr(instr_set1)
            p.add_instr(instr_set2)
        return p


class Program:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.gates = [] 

    def add_instr(self, gate_list):
        for gate in gate_list:
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










###########################################    Main Functionality (Heisenberg Hamiltonian Circuit Generation and Evaluation)  ##################################################################################################
class Heisenberg:

    def __init__(self,file="input_file.txt"):
        #%matplotlib inline 
        input_file=open(file,'r')
        data=input_file.readlines()
        self.logfile=open(completename,'w')



        self.H_BAR = 0.658212    # eV*fs

        #Default Parameters
        self.JX=self.JY=self.JZ=self.h_ext=0
        self.ext_dir="Z"
        self.num_qubits=2
        self.initial_spins="1,1"
        self.delta_t=1
        self.steps=1
        self.QCQS="QS"
        self.shots=1024
        self.noise_choice="n"
        self.device_choice="ibmq_rome"
        self.plot_flag="y"
        self.freq=0
        self.time_dep_flag="n"
        self.custom_time_dep="n"
        self.print_bool=0 #controls print statements for smart compiler integration
        self.smart_bool=False #controls transpiling in presence of smart compilation
        self.circuits_list=[]
        self.backend="ibm"
        self.ibm_circuits_list=[]
        self.rigetti_circuits_list=[]
        self.cirq_circuits_list=[]
        self.auto_smart_compile="y"
        self.default_compiler-"native" #native or smart

        from numpy import cos as cos_func
        self.time_func=cos_func

        for i in range(len(data)-1):
            value=data[i+1].strip()
            if "*JX" in data[i]:
                self.JX=float(value)
            elif "*JY" in data[i]:
                self.JY=float(value)
            elif "*JZ" in data[i]:
                self.JZ=float(value)
            elif "*h_ext" in data[i]:
                self.h_ext=float(value)
            elif "*initial_spins" in data[i]:
                self.initial_spins=value
            elif "*delta_t" in data[i]:
                self.delta_t=int(value)
            elif "*steps" in data[i]:
                self.steps=int(value)
            elif "*num_qubits" in data[i]:
                self.num_qubits=int(value)
            elif "*QCQS" in data[i]:
                self.QCQS=value
            elif "*device" in data[i]:
                self.device_choice=value
            elif "*backend" in data[i]:
                self.backend=value
            elif "*noise_choice" in data[i]:
                self.noise_choice=value
            elif "*plot_flag" in data[i]:
                self.plot_flag=value
            elif "*shots" in data[i]:
                self.shots=int(value)
            elif "*freq" in data[i]:
                self.freq=float(value)
            elif "*time_dep_flag" in data[i]:
                self.time_dep_flag=value
            elif "*default_compiler" in data[i]:
                self.default_compiler=value
            elif "*ext_dir" in data[i]:
                self.ext_dir=value
            elif "*auto_smart_compile" in data[i]:
                self.auto_smart_compile=value
            elif "*custom_time_dep" in data[i]:
                self.custom_time_dep=value
                if self.custom_time_dep in "y":
                    from time_dependence import external_func
                    print("Found an external time dependence function")
                    self.logfile.write("Found an external time dependence function")
                    self.time_func=external_func


        if "y" in self.plot_flag:
            import matplotlib.pyplot as plt


        self.initial_spins=self.initial_spins.split(',')


        self.total_time=int(self.delta_t*self.steps)



        self.flip_vec=np.zeros(self.num_qubits)
        index=0
        for spin in self.initial_spins:
            if int(spin)==-1:
                self.flip_vec[index]=1
                index+=1
            elif int(spin)==1:
                self.flip_vec[index]=0
                index+=1
            else: 
                print('Invalid spin entered')
                self.logfile.write("Invalid spin entered\n")

        if self.backend in "ibm":
            ## Declare registers
            self.qr = qk.QuantumRegister(self.num_qubits, 'q')
            self.cr = qk.ClassicalRegister(self.num_qubits, 'c')




    def local_evolution_circuit(self,evol_time): #creates evolution circuit in local program
    #Initial flipped spins are not implemented in this function due to the need for "barrier". Need to do that outside of this.
        prop_steps = int(evol_time / self.delta_t)  # number of propagation steps
        P=program(self,num_qubits)
        for t in range(prop_steps):
            t = (step + 0.5) * self.delta_t
            if "n" in self.time_dep_flag:
                psi_ext = -2.0 * self.h_ext *self.delta_t / self.H_BAR
            elif "y" in self.time_dep_flag:
                if "y" in self.custom_time_dep:
                    psi_ext = -2.0 * self.h_ext * self.time_func(t)*self.delta_t / self.H_BAR
                elif "n" in self.custom_time_dep:
                    psi_ext=-2.0*self.h_ext*np.cos(self.freq*t)*self.delta_t/self.H_BAR
                else:
                    print("Invalid selection for custom_time_dep parameter. Please enter y or n.")
                    self.logfile.write("Invalid selection for custom_time_dep parameter. Please enter y or n.\n")
                    break
            ext_instr_set=[]
            XX_instr_set=[]
            YY_instr_set=[]
            ZZ_instr_set=[]
            for q in range(self.num_qubits):
                if self.ext_dir in "X":
                    ext_instr_set.append(Gate('H', [q]))
                    ext_instr_set.append(Gate('RZ', [q], angles=[psi_ext]))
                    ext_instr_set.append(Gate('H',[q]))
                elif self.ext_dir in "Y":
                    ext_instr_set.append(Gate('RX', [q], angles=[-np.pi/2]))
                    ext_instr_set.append(Gate('RZ', [q], angles=[psi_ext]))
                    ext_instr_set.append(Gate('RX', [q], angles=[np.pi/2]))
                elif self.ext_dir in "Z":
                    ext_instr_set.append(Gate('RZ', [q], angles=[psi_ext]))

            psiX=-2.0*(self.JX)*self.delta_t/self.H_BAR
            psiY=-2.0*(self.JY)*self.delta_t/self.H_BAR
            psiZ=-2.0*(self.JZ)*self.delta_t/self.H_BAR

            for q in range(self.num_qubits-1):
                XX_instr_set.append(Gate('H',[q]))
                XX_instr_set.append(Gate('H',[q+1]))
                XX_instr_set.append(Gate('CNOT',[q, q+1]))
                XX_instr_set.append(Gate('RZ', [q], angles=[psiX]))
                XX_instr_set.append(Gate('CNOT',[q, q+1]))
                XX_instr_set.append(Gate('H',[q]))
                XX_instr_set.append(Gate('H',[q+1]))

                YY_instr_set.append(Gate('RX',[q],angles=[-np.pi/2]))
                YY_instr_set.append(Gate('RX',[q+1],angles=[-np.pi/2]))
                YY_instr_set.append(Gate('CNOT',[q, q+1]))
                YY_instr_set.append(Gate('RZ', [q], angles=[psiY]))
                YY_instr_set.append(Gate('CNOT',[q, q+1]))
                YY_instr_set.append(Gate('RX',[q],angles=[np.pi/2]))
                YY_instr_set.append(Gate('RX',[q+1],angles=[np.pi/2]))

                ZZ_instr_set.append(Gate('CNOT',[q, q+1]))
                ZZ_instr_set.append(Gate('RZ', [q], angles=[psiZ]))
                ZZ_instr_set.append(Gate('CNOT',[q, q+1]))

            P.add_instr(ext_instr_set)
            P.add_instr(XX_instr_set)
            P.add_instr(YY_instr_set)
            P.add_instr(ZZ_instr_set)

    def generate_local_circuits(self):

        ## Create circuits
        circuits = []
        for j in range(0, self.steps+1):
            print("Generating timestep {} circuit".format(j))
            evolution_time = self.delta_t * j
            circuits.append(self.local_evolution_circuit(self,evolution_time))

        self.circuits_list=circuits


    def generate_ibm(self):
        #convert from local circuits to IBM-specific circuit

        ## Show available backends
        provider = qk.IBMQ.get_provider(group='open')
        provider.backends()

        #choose the device you would like to run on
        device = provider.get_backend(self.device_choice)

        #gather fidelity statistics on this device if you want to create a noise model for the simulator
        properties = device.properties()
        coupling_map = device.configuration().coupling_map

        #TO RUN ON THE SIMULATOR 
        #create a noise model to use for the qubits of the simulator
        noise_model = NoiseModel.from_backend(device)
        # Get the basis gates for the noise model
        basis_gates = noise_model.basis_gates

        # Select the QasmSimulator from the Aer provider
        simulator = Aer.get_backend('qasm_simulator')


        #To run on the quantum computer, assign a quantum computer of your choice as the backend 
        backend = provider.get_backend(self.device_choice)


        print("Creating IBM quantum circuit objects...")
        self.logfile.write("Creating IBM quantum circuit objects...")
        for circuit in self.circuits_list:
            circ = qk.QuantumCircuit(self.qr, self.cr)
            index=0
            for flip in self.flip_vec:
                if int(flip)==1:
                    circ.x(self.qr[index])
                    index+=1
                else: index+=1
            circ.barrier()
            for gate in circuit.gates:
                if gate.name in "H":
                    circ.h(self.qubits)
                elif gate.name in "RZ":
                    circ.rz(self.angles,self.qubits)
                elif gate.name in "RX":
                    circ.rx(self.angles,self.qubits)
                elif gate.name in "CNOT":
                    circ.cx(self.qubits)
            circ.measure(self.qr,self.cr)
            self.ibm_circuits_list.append(circ)
        print("IBM quantum circuit objects created")
        self.logfile.write("IBM quantum circuit objects created")

        if self.JZ != 0 and self.JX==self.JY==0 and self.h_ext!=0 and self.ext_dir=="X" and self.auto_smart_compile=="y":
            #TFIM
            print("TFIM detected, enabling smart compiler")
            self.logfile.write("TFIM detected, enabling smart compiler")
            temp=[]
            for circuit in self.ibm_circuits_list:
                temp.append(smart_compile(circuit))
            self.ibm_circuits_list=temp

        elif self.default_compiler in "smart":
            temp=[]
            for circuit in self.ibm_circuits_list:
                temp.append(smart_compile(circuit))
            self.ibm_circuits_list=temp
        elif self.default_compiler in "native":
            temp=[]
            for circuit in self.ibm_circuits_list:
                circ = qk.transpile(circuits, backend=backend, optimization_level=3)
                temp.append(circ)
            self.ibm_circuits_list=temp

        return ibm_circuits_list
    def connect_account(self,api_key=None, overwrite=False):
        if api_key != None:
            if overwrite==False:
                qk.IBMQ.save_account(api_key) ## only run once!
            else:
                qk.IBMQ.save_account(api_key,overwrite=True) ## only run once!
     #qk.IBMQ.delete_accounts() ## only run if you need to use a new token
        qk.IBMQ.load_account()


    def parameters(self):
        print("Current model parameters:\n\nH_BAR = {}\nJX = {}\nJY = {}\nJZ = {}\nh_ext = {}\next_dir = {}".format(self.H_BAR,self.JX,self.JY,self.JZ,self.h_ext,self.ext_dir))
        print("num_qubits = {}\ninitial_spins = {}\ndelta_t = {}\nsteps = {}\nQCQS = {}\nshots = {}\nnoise_choice = {}".format(self.num_qubits,self.initial_spins,self.delta_t,self.steps,self.QCQS,self.shots,self.noise_choice))
        print("device choice = {}\nplot_flag = {}\nfreq = {}\ntime_dep_flag = {}\ncustom_time_dep = {}\n".format(self.device_choice,self.plot_flag,self.freq,self.time_dep_flag,self.custom_time_dep)) 

    def results(self):
        return self.result_matrix

    def return_circuits(self):
        return self.circuits_list

    def average_magnetization(self,result: dict, shots: int, qub: int):
      """Compute average magnetization from results of qk.execution.
      Args:
      - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
      - shots (int): number of trials
      Return:
      - average_mag (float)
      """
      mag = 0
      for spin_str, count in result.items():
        spin_int = [1 - 2 * float(spin_str[qub])]
        #print(spin_str)
        mag += (sum(spin_int) / len(spin_int)) * count
      average_mag = mag / shots
      return average_mag














    def run_circuits(self):
        ## Show available backends
        provider = qk.IBMQ.get_provider(group='open')
        provider.backends()

        #choose the device you would like to run on
        device = provider.get_backend(self.device_choice)

        #gather fidelity statistics on this device if you want to create a noise model for the simulator
        properties = device.properties()
        coupling_map = device.configuration().coupling_map

        #TO RUN ON THE SIMULATOR 
        #create a noise model to use for the qubits of the simulator
        noise_model = NoiseModel.from_backend(device)
        # Get the basis gates for the noise model
        basis_gates = noise_model.basis_gates

        # Select the QasmSimulator from the Aer provider
        simulator = Aer.get_backend('qasm_simulator')


        #To run on the quantum computer, assign a quantum computer of your choice as the backend 
        backend = provider.get_backend(self.device_choice)

        #CHOOSE TO RUN ON QUANTUM COMPUTER OR SIMULATOR
        if self.QCQS in ["QC"]:
            #quantum computer execution
            job = qk.execute(self.circuits_list, backend=backend, shots=self.shots)
            job_monitor(job)

        elif self.QCQS in ["QS"]:
            #simulator execution
            if self.noise_choice in ["y"]:
                print("Running noisy simulator job...")
                self.logfile.write("Running noisy simulator job...\n")
                result_noise = execute(self.circuits_list, simulator, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,shots=self.shots).result()
                print("Noisy simulator job successful")
            elif self.noise_choice in ["n"]:
                print("Running noiseless simulator job...")
                self.logfile.write("Running noiseless simulator job...\n")
                result_noise=execute(self.circuits_list,simulator,coupling_map=coupling_map,basis_gates=basis_gates,shots=self.shots).result()
                print("Noiseless simulator job successful")
                self.logfile.write("Noiseless simulator job successful")
            else: 
                print("Please enter either y or n for the simulator noise query")
                self.logfile.write("Please enter either y or n for the simulator noise query\n")
        else:
            print("Please enter either QC or QS")
            self.logfile.write("Please enter either QC or QS\n")


        
            


        #Post Processing Depending on Choice
        self.result_out_list=[]
        if self.QCQS in ["QS"]:
            #SIMULATOR POST PROCESSING
            for j in range(self.num_qubits):
                avg_mag_sim = []
                temp = []
                i = 1
                print("Post-processing qubit {} data".format(j+1))
                self.logfile.write("Post-processing qubit {} data\n".format(j+1))
                for c in self.circuits_list:
                    result_dict = result_noise.get_counts(c)
                    temp.append(self.average_magnetization(result_dict, self.shots,j))
                    if i % self.steps == 0:
                        avg_mag_sim.append(temp)
                        temp = []
                    i += 1
                # time_vec=np.linspace(0,total_t,steps)
                # time_vec=time_vec*JX/H_BAR
                if "y" in self.plot_flag:
                    plt.figure()
                    plt.plot(range(self.steps), avg_mag_sim[0])
                    plt.xlabel("Simulation Timestep")
                    plt.ylabel("Average Magnetization")
                    plt.savefig("Data/Simulator_result_qubit{}.png".format(j+1))
                    plt.close()
                self.result_out_list.append(avg_mag_sim[0])
                np.savetxt("Data/Qubit {} Average Magnetization Data.txt".format(j+1),avg_mag_sim[0])
            self.result_matrix=np.stack(self.result_out_list)
            print("Done")
            self.logfile.write("Done\n")

        elif self.QCQS in ["QC"]:
            #QUANTUM COMPUTER POST PROCESSING
            for j in range(self.num_qubits):
                results = job.result()        
                avg_mag_qc = []
                temp = []
                i = 1
                print("Post-processing qubit {} data".format(j+1))
                self.logfile.write("Post-processing qubit {} data\n".format(j+1))
                for c in self.circuits_list:
                        result_dict = results.get_counts(c)
                        temp.append(self.average_magnetization(result_dict, self.shots,j))
                        if i % self.steps == 0:
                                avg_mag_qc.append(temp)
                                temp = []
                        i += 1
                
                # QC
                if "y" in self.plot_flag:
                    plt.figure()
                    plt.plot(range(self.steps), avg_mag_qc[0])
                    plt.xlabel("Simulation Timestep")
                    plt.ylabel("Average Magnetization")
                    plt.savefig("Data/QC_result_qubit{}.png".format(j+1))
                    plt.close()
                self.result_out_list.append(avg_mag_qc[0])
                np.savetxt("Data/Qubit {} Average Magnetization Data.txt".format(j+1),avg_mag_qc[0])
            self.result_matrix=np.stack(self.result_out_list)           
            print("Done")
            self.logfile.write("Done\n")

















############    Pure Smart Compiler Functionality   #######################################################################################################################################
#If the user just wants to pass an existing circuit object through the smart compilers
def smart_compile(circ_obj):
    if isintance(circ_obj,qiskit.circuit.quantumcircuit.QuantumCircuit): #IBM case
        nqubits=circ_obj.num_qubits    
        #Read the gate in right vector form
        # G = Gate type
        # TH = Angle of rotation ! if no angle rotation then TH = 0
        # TH2 = 2nd angle of rotation (used in U2 and U3 gates)
        # TH3 = 3rd angle of rotation (used in U3 gates)
        # AC1 = qubit on which action is happening
        # AC2 = qubit on which controlled action is happening
        instr_list=circ_obj.data
        count = len(instr_list)
     
        G = ["" for x in range(count)]
        G = list(G)
        AC1 = np.zeros(shape=(count),dtype=np.int) 
        AC2 = np.zeros(shape=(count),dtype=np.int) 
        TH = np.zeros(shape=(count))
        i = 0
        for instr in instr_list:
            G[i] = 0
            TH[i] = 0
            AC1[i] = 0
            AC2[i] = 0
            name = instr[0].name
            if name == "h":
                G[i]="H"
                TH[i] = 0
                AC1[i] = instr[1][0].index
                AC2[i] = 0
            if name == "rz":
                G[i] = "RZ"
                TH[i] = instr[0].params[0]
                AC1[i] = instr[1][0].index
                AC2[i] = 0
            if name == "cx":
                G[i] = "CNOT"
                TH[i] = 0
                AC1[i] = instr[1][0].index
                AC2[i] = instr[1][1].index
            if name == "measure":
                G[i] = "MEASURE"
                TH[i] = 0
                AC1[i] = 0
                AC2[i] =0
            i = i+1
            
            
            
        #Omit last and second-to-last CNOT for each qubit
        for qub in range(0,nqubits+1):
            i=-1
            count=0
            while count<=1 and i>=-int(len(G)):
                if G[i] == "CNOT" and AC1[i]==qub and AC2[i]==qub+1:
                    del G[i]
                    TH=np.delete(TH,i)
                    AC1=np.delete(AC1,i)
                    AC2=np.delete(AC2,i)
                    count=count+1
                i=i-1

        #Omit last RZ for each qubit
        for qub in range(0,nqubits+1):
            i=-1
            while i>=-int(len(G)):
                if G[i] == "H" and AC1[i]==qub:
                    break
                if G[i] == "RZ" and AC1[i]==qub:
                    G[i]  = "NULL"
                    break
                i=i-1            
                
                
        #Use CNOT (0,1) ->  H(0) H(1) CNOT(1,0) H(0) H(1)
        i=0
        while G[i] != "MEASURE":
            if G[i]=="CNOT" and (G[i+1]=="H" and G[i+2]=="H" and AC1[i+1]==AC1[i] and AC1[i+2]==AC2[i])==False:
                G[i]="H"
                flag1=int(AC1[i])
                flag2=int(AC2[i])
                AC2[i]=0
                G.insert(i,"H")
                TH=np.insert(TH,i,0)
                AC1=np.insert(AC1,i,flag2)
                AC2=np.insert(AC2,i,0)
                G.insert(i,"CNOT")
                TH=np.insert(TH,i,0)
                AC1=np.insert(AC1,i,flag2)
                AC2=np.insert(AC2,i,flag1)
                G.insert(i,"H")
                TH=np.insert(TH,i,0)
                AC1=np.insert(AC1,i,flag1)
                AC2=np.insert(AC2,i,0)
                G.insert(i,"H")
                TH=np.insert(TH,i,0)
                AC1=np.insert(AC1,i,flag2)
                AC2=np.insert(AC2,i,0)
            i=i+1

        #Rearrange circuits to put successive Hadamard gates in order
        i=0
        while G[i] != "MEASURE":
            if G[i]=="H":
                flag=AC1[i]
                j=i+1
                boolean=0
                while G[j] != "MEASURE" and boolean ==0:
                    if AC1[j]==flag and G[j] == "H":
                        boolean=1
                        del G[j]
                        TH=np.delete(TH,j)
                        AC1=np.delete(AC1,j)
                        AC2=np.delete(AC2,j)
                        G.insert(i,"H")
                        TH=np.insert(TH,i,0)
                        AC1=np.insert(AC1,i,flag)
                        AC2=np.insert(AC2,i,0)
                    if AC1[j]==flag and G[j] != "H":
                        break
                    j=j+1
            i=i+1

      
        #Use successive Hadamard annihilation
        i=0
        while G[i]!= "MEASURE":
            if G[i]=="H" and G[i+1] == "H" and AC1[i]==AC1[i+1]:
                del G[i]
                TH=np.delete(TH,i)
                AC1=np.delete(AC1,i)
                AC2=np.delete(AC2,i)
                del G[i]
                TH=np.delete(TH,i)
                AC1=np.delete(AC1,i)
                AC2=np.delete(AC2,i)
                i=i-1
            i=i+1
            
            
        #Convert HRZ(theta)H to RZ(pi/2)RX(pi/2)RZ(theta+pi)RX(pi/2)RZ(pi/2)
        i=0
        while G[i] != "MEASURE":
            if (G[i] == "H" and G[i+1] == "RZ" and G[i+2]=="H" and AC1[i] == AC1[i+1] and AC1[i+1]== AC1[i+2]):
                theta = TH[i+1]
                q = AC1[i]
                G[i]="RZ"
                TH[i]=1.57079632679
                del G[i+1]
                TH=np.delete(TH,i+1)
                AC1=np.delete(AC1,i+1)
                AC2=np.delete(AC2,i+1)
                del G[i+1]
                TH=np.delete(TH,i+1)
                AC1=np.delete(AC1,i+1)
                AC2=np.delete(AC2,i+1)
                G.insert(i,"RX")
                TH=np.insert(TH,i,1.57079632679)
                AC1=np.insert(AC1,i,q)
                AC2=np.insert(AC2,i,q)
                G.insert(i,"RZ")
                TH=np.insert(TH,i,theta+(2.0*1.57079632679))
                AC1=np.insert(AC1,i,q)
                AC2=np.insert(AC2,i,q)
                G.insert(i,"RX")
                TH=np.insert(TH,i,1.57079632679)
                AC1=np.insert(AC1,i,q)
                AC2=np.insert(AC2,i,q)
                G.insert(i,"RZ")
                TH=np.insert(TH,i,1.57079632679)
                AC1=np.insert(AC1,i,q)
                AC2=np.insert(AC2,i,q)

                #move leftmost RZ of set across control bit if possible
                for j in range(i-1,0,-1):
                    if AC1[j] == AC1[i]:
                        if G[j] == "CNOT":
                            for k in range(j-1,0,-1):
                                if AC1[k] == AC1[i]:
                                    if G[k] == "RZ":
                                        TH[k]=TH[k]+TH[i]
                                        del G[i]
                                        TH=np.delete(TH,i)
                                        AC1=np.delete(AC1,i)
                                        AC2=np.delete(AC2,i)
                                    else: break
                        else: break
                            
                #move rightmost RZ of set across control bit if possible
                for j in range(i+4,len(G)):
                    if AC1[j] == AC1[i+3]:
                        if G[j] == "CNOT":
                            for k in range(j+1,len(G)):
                                if AC1[k] == AC1[i+3]:
                                    if G[k] == "RZ":
                                        TH[k]=TH[k]+TH[i+3]
                                        del G[i+3]
                                        TH=np.delete(TH,i+3)
                                        AC1=np.delete(AC1,i+3)
                                        AC2=np.delete(AC2,i+3)
                                    else: break
                                if AC2[k] == AC1[i+3]:
                                    break
                        else: break
                            
                            
            i=i+1

        #convert remaining HRZ or H to native gates
        i=0
        while G[i] != "MEASURE":
            if G[i]=="H":
                q = AC1[i]
                j=i+1
                flag = 1
                while G[j] != "MEASURE":
                    if AC1[j] == AC1[i]:
                        #change HRZ to native gates
                        if G[j]=="RZ":
                            G[i] = "RZ"
                            theta = TH[j]
                            TH[i]=1.57079632679
                            del G[j]
                            TH=np.delete(TH,j)
                            AC1=np.delete(AC1,j)
                            AC2=np.delete(AC2,j)
                            G.insert(i+1,"RX")
                            TH=np.insert(TH,i+1,1.57079632679)
                            AC1=np.insert(AC1,i+1,q)
                            AC2=np.insert(AC2,i+1,0)
                            G.insert(i+2,"RZ")
                            TH=np.insert(TH,i+2,theta+1.57079632679)
                            AC1=np.insert(AC1,i+2,q)
                            AC2=np.insert(AC2,i+2,0)
                            flag = 0
                            break
                        else: break
                    j=j+1
                #change H to native gates    
                if (flag):
                    G[i] = "RZ"
                    TH[i]=1.57079632679
                    G.insert(i+1,"RX")
                    TH=np.insert(TH,i+1,1.57079632679)
                    AC1=np.insert(AC1,i+1,q)
                    AC2=np.insert(AC2,i+1,0)
                    G.insert(i+2,"RZ")
                    TH=np.insert(TH,i+2,1.57079632679)
                    AC1=np.insert(AC1,i+2,q)
                    AC2=np.insert(AC2,i+2,0)  
                #compress successive RZs
                if (G[i-1] == "RZ" and AC1[i-1] == AC1[i]):
                    TH[i-1] = TH[i-1]+TH[i]
                    del G[i]
                    TH=np.delete(TH,i)
                    AC1=np.delete(AC1,i)
                    AC2=np.delete(AC2,i)
                #if (G[i+3] == "RZ"):
                #    TH[i+2] = TH[i+2]+TH[i+3]
                #    del G[i+3]
                #    TH=np.delete(TH,i+3)
                #    AC1=np.delete(AC1,i+3)
                #    AC2=np.delete(AC2,i+3)
               
            i=i+1
            
        #Omit first RZs
        for qub in range(0,nqubits):
            i=0
            while G[i] != "MEASURE":
                if G[i]=="RZ" and AC1[i]==qub:
                    del G[i]
                    TH=np.delete(TH,i)
                    AC1=np.delete(AC1,i)
                    AC2=np.delete(AC2,i)
                if (G[i]=="RX" and AC1[i]==qub) or (G[i]=="CNOT" and (AC1[i]==qub or AC2[i]==qub)):
                    break
                i=i+1
            
        #Omit last RZ for each qubit
        for qub in range(0,nqubits+1):
            i=-1
            while i>=-int(len(G)):
                if G[i] == "H" and AC1[i]==qub:
                    break
                if G[i] == "RZ" and AC1[i]==qub:
                    G[i]  = "NULL"
                    break
                i=i-1  
      
        #build output circuit
        qr = qk.QuantumRegister(nqubits, 'q')
        cr = qk.ClassicalRegister(nqubits, 'c')
        
        circuit = qk.QuantumCircuit(qr, cr)

        for i in range(len(G)):
            if (G[i] == "RX"):
                circuit.rx(TH[i], int(AC1[i]))
            if (G[i] == "RZ"):
                circuit.rz(TH[i], int(AC1[i]))
            if (G[i] == "CNOT"):
                circuit.cx(int(AC1[i]), int(AC2[i]))
            if (G[i] == "H"):
                circuit.h(int(AC1[i]))


        circuit.measure(qr, cr)
        
        return circuit













    ##################    Relic Zone    ################################################################################################################

        # def evolution_circuit(self, evol_time,circ_name=None):      #old IBM-centric version
    #         """
    #         Define circuit for evolution of wavefunction, i.e.,
    #         H(t) = - Jz * sum_{i=1}^{N-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
    #                      - e_ph * cos(w_ph * t) * sum_{i=1}^{N}(sigma_{x}^{i})
            
    #         Returns:
    #         - A quantum circuit representating the propagation of the system.
    #         """
            
    #         assert self.num_qubits == self.qr.size    
    #         prop_steps = int(evol_time / self.delta_t)  # number of propagation steps
            
    #         # Instantiate quantum circuit for the propagator to which
    #         #-we add terms of Hamiltonian piece by piece
    #         circuit = qk.QuantumCircuit(self.qr, self.cr)
    #         if circ_name:
    #                 circuit.name = circ_name
    #         index=0
    #         for flip in self.flip_vec:
    #             if int(flip)==1:
    #                 circuit.x(self.qr[index])
    #                 index+=1
    #             else: index+=1
    #         circuit.barrier()



    #         for step in range(prop_steps):
    #             t = (step + 0.5) * self.delta_t


    #             if "n" in self.time_dep_flag:
    #                 psi_ext = -2.0 * self.h_ext *self.delta_t / self.H_BAR
    #             elif "y" in self.time_dep_flag:
    #                 if "y" in self.custom_time_dep:
    #                     psi_ext = -2.0 * self.h_ext * self.time_func(t)*self.delta_t / self.H_BAR
    #                 elif "n" in self.custom_time_dep:
    #                     psi_ext=-2.0*self.h_ext*np.cos(self.freq*t)*self.delta_t/self.H_BAR
    #                 else:
    #                     print("Invalid selection for custom_time_dep parameter. Please enter y or n.")
    #                     self.logfile.write("Invalid selection for custom_time_dep parameter. Please enter y or n.\n")
    #                     break
    #             prop_circ = qk.QuantumCircuit(self.qr)

    #             if self.ext_dir in "Z":
    #                 #Z-Direction External Field Term
    #                 prop_circ.rz(psi_ext,self.qr)
    #             elif self.ext_dir in "X":
    #                 #X-Direction External Field Term
    #                 prop_circ.h(self.qr)
    #                 prop_circ.rz(psi_ext,self.qr)
    #                 prop_circ.h(self.qr)

    #             elif self.ext_dir in "Y":
    #                 #Y-Direction External Field Term
    #                 prop_circ.rx(-np.pi/2,self.qr)
    #                 prop_circ.rz(psi_ext,self.qr)
    #                 prop_circ.rx(np.pi/2,self.qr)

    #             #XX Coupling Term
    #             psi2=-2.0*(self.JX)*self.delta_t/self.H_BAR
    #             for i in range(self.num_qubits-1):
    #                 prop_circ.h(self.qr[i])
    #                 prop_circ.h(self.qr[i+1])
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #                 prop_circ.rz(psi2, self.qr[i+1])
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #                 prop_circ.h(self.qr[i])
    #                 prop_circ.h(self.qr[i+1])
    #             #YY Coupling Term
    #             psi2=-2.0*(self.JY)*self.delta_t/self.H_BAR
    #             for i in range(self.num_qubits-1):
    #                 prop_circ.rx(-np.pi/2,self.qr[i])
    #                 prop_circ.rx(-np.pi/2,self.qr[i+1])
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #                 prop_circ.rz(psi2, self.qr[i+1])
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #                 prop_circ.rx(np.pi/2,self.qr[i])
    #                 prop_circ.rx(np.pi/2,self.qr[i+1])
    #             # ZZ Coupling Term
    #             psi2 = -2.0 * self.JZ * self.delta_t / self.H_BAR
    #             for i in range(self.num_qubits-1):
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #                 prop_circ.rz(psi2, self.qr[i+1])
    #                 prop_circ.cx(self.qr[i], self.qr[i+1])
    #             # Concatenate circuites
    #             circuit += prop_circ
    #             #circuit.barrier(qr)
    #         # Add measurement operation
    #         circuit.measure(self.qr, self.cr)

            # if self.JZ != 0 and self.JX==self.JY==0:
            #     from smart_compile_native import smart_compile
            #     if self.h_ext != 0:
            #         if self.ext_dir in "X":
            #             if self.print_bool==0:
            #                 print("Ising model detected, smart compiler enabled")
            #                 self.logfile.write("Ising model detected, smart compiler enabled")
            #                 self.print_bool=1
            #             circuit=smart_compile(circuit.data,self.num_qubits)
            #             self.smart_bool=True
            #     else:
            #         if self.print_bool==0:
            #             print("Ising model detected, smart compiler enabled")
            #             self.logfile.write("Ising model detected, smart compiler enabled")
            #             self.print_bool=1
            #         circuit=smart_compile(circuit.data,self.num_qubits)
            #         self.smart_bool=True

            # return circuit