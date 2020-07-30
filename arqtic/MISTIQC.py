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

import matplotlib.pyplot as plt
import os



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


        if self.backend in "ibm":
            import qiskit as qk
            from qiskit.tools.monitor import job_monitor
            from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
            from qiskit import Aer, IBMQ, execute
            from qiskit.providers.aer import noise
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit.circuit import quantumcircuit
            from qiskit.circuit import Instruction
        elif self.backend in "rigetti":
            import pyquil
            from pyquil.quil import Program
            from pyquil.gates import RX, RZ, CZ, RESET, MEASURE
        elif self.backend in "cirq":
            import cirq

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
                temp.append(smart_compile(circuit,self.backend))
            self.ibm_circuits_list=temp

        elif self.default_compiler in "smart":
            temp=[]
            print("Compiling circuits...")
            self.logfile.write("Compiling circuits...")
            for circuit in self.ibm_circuits_list:
                temp.append(smart_compile(circuit,self.backend))
            self.ibm_circuits_list=temp
            print("Circuits compiled successfully")
            self.logfile.write("Circuits compiled successfully")
        elif self.default_compiler in "native":
            temp=[]
            print("Transpiling circuits...")
            self.logfile.write("Transpiling circuits...")
            for circuit in self.ibm_circuits_list:
                circ = qk.transpile(circuits, backend=backend, optimization_level=3)
                temp.append(circ)
            self.ibm_circuits_list=temp
            print("Circuits transpiled successfully")
            self.logfile.write("Circuits transpiled successfully")


    def generate_rigetti(self):
        print("Creating Pyquil program list...")
        self.logfile.write("Creating Pyquil program list...")
        for circuit in self.circuits_list:
            p = Program(RESET()) #compressed program
            ro = p.declare('ro', memory_type='BIT', memory_size=self.num_qubits)
            for gate in circuit.gates:
                if gate.name in "H":
                    p.inst(H(self.qubits))
                elif gate.name in "RZ":
                    p.inst(RZ(self.angles,self.qubits))
                elif gate.name in "RX":
                    p.inst(RX(self.angles,self.qubits))
                elif gate.name in "CNOT":
                    p.inst(CZ(self.qubits))
            for i in range(self.num_qubits):
                p.inst(MEASURE(i,ro[i]))
            rigetti_circuits_list.append(p)
        print("Pyquil program list created successfully")
        self.logfile.write("Pyquil program list created successfully")

    def generate_cirq(self):
        print("Creating Cirq circuit list...")
        self.logfile.write("Creating Cirq circuit list...")
        for circuit in self.circuits_list:
            c=cirq.Circuit()
            qubit_list=cirq.LineQubit.range(self.num_qubits)
            gate_list=[]
            for gate in circuit.gates:
                if gate.name in "H":
                    gate_list.append(cirq.H(qubit_list[self.qubits]))
                elif gate.name in "RZ":
                    gate_list.append(cirq.rz(self.angles)(qubit_list[self.qubits]))
                elif gate.name in "RX":
                    gate_list.append(cirq.rx(self.angles)(qubit_list[self.qubits]))
                elif gate.name in "CNOT":
                    gate_list.append(cirq.CNOT(qubit_list[self.qubits]))
            gate_list.append(cirq.measure(qubit_list))
            c.append(gate_list,strategy=InsertStrategy.EARLIEST)
            cirq_circuits_list.append(c)
        print("Successfully created Cirq circuit list")
        self.logfile.write("Successfully created Cirq circuit list")

    def generate_circuits(self):
        self.generate_local_circuits()
        if self.backend in "ibm":
            self.generate_ibm()
        if self.backend in "rigetti":
            self.generate_rigetti()
        if self.backend in "cirq":
            self.generate_cirq()

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
        #this is missing some of the latest parameter additions

    def results(self):
        return self.result_matrix

    def return_circuits(self):
        if self.backend in "ibm":
            return self.ibm_circuits_list
        elif self.backend in "rigetti":
            return self.rigetti_circuits_list
        elif self.backend in "cirq":
            return self.cirq_circuits_list

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
        if self.backend in "ibm":
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
def smart_compile(circ_obj,circ_type):
    if circ_type in "qiskit":
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


    elif circ_type in "rigetti":
        nqubits=len(circ_obj.get_qubits())
        lineList = [str(instr) for instr in circ_obj]
        count = len(lineList)
        
        #Read the gate in right vector form
        # G = Gate type
        # TH = Angle of rotation ! if no angle rotation then TH = 0
        # AC1 = qubit on which action is happening
        # AC2 = qubit on which controlled action is happening

        G = ["" for x in range(count)]
        G = list(G)
        AC1 = np.zeros(shape=(count),dtype=np.int) 
        AC2 = np.zeros(shape=(count),dtype=np.int) 
        TH = np.zeros(shape=(count))
        for i in range (0,count):
          G[i] = 0
          TH[i] = 0
          AC1[i] = 0
          AC2[i] = 0
          if lineList[i][0:1] == "H":
            G[i]="H"
            TH[i] = 0
            AC1[i] = lineList[i][2:3]
            AC2[i] = 0
          if lineList[i][0:2] == "RZ":
            G[i] = "RZ"
            TH[i] = lineList[i][lineList[i].find("(")+1:lineList[i].find(")")]
            AC1[i] = lineList[i][-1]
            AC2[i] = 0
          if lineList[i][0:4] == "CNOT":
            G[i] = "CNOT"
            TH[i] = 0
            AC1[i] = lineList[i][5:6] 
            AC2[i] = lineList[i][7:8]
          if lineList[i][0:7] == "MEASURE":
            G[i] = "MEASURE"
            TH[i] = 0
            AC1[i] = 0
            AC2[i] =0
        
        #qiskit_code(G,TH,AC1,AC2,"qiskit_uncompressed.txt")
        #rigetti_code(G,TH,AC1,AC2,"rigetti_uncompressed.txt")

     
        # Use CNOT = H CZ H 
        i = 0
        while G[i] != "MEASURE":
          if G[i] == "CNOT":
             G[i] = "CZ"
             G.insert(i+1,"H")
             TH = np.insert(TH,i+1,0) 
             AC1 = np.insert(AC1,i+1,AC2[i]) 
             AC2 = np.insert(AC2,i+1,0)
             G.insert(i,"H")
             TH = np.insert(TH,i,0) 
             AC1 = np.insert(AC1,i,AC2[i]) 
             AC2 = np.insert(AC2,i,0)
          i = i+1  
        
        # Last and second last CNOT can be ommited  
        maxq = max(max(AC1),max(AC2))
        remember = np.zeros(shape=(2,maxq),dtype=np.int)
        for mm in range (0,maxq+1):
         i = 0
         while G[i] != "MEASURE":
              if G[i] == "CZ" and AC1[i] == mm and AC2[i] == mm+1:
                    j = i+1
                    while G[j] != "MEASURE":
                          if G[j] == "CZ" and AC1[j] == mm and AC2[j] == mm+1:
                               remember[0][mm] = i; remember[1][mm] = j;
                          j = j+1
              i = i+1 
        
        for nn in range (maxq-1,-1,-1):
         for mm in range (1,-1,-1):
        #   print(mm,nn)
          del G[remember[mm][nn]];TH = np.delete(TH,remember[mm][nn]);
          AC1 = np.delete(AC1,remember[mm][nn]); AC2 = np.delete(AC2,remember[mm][nn])
        
        
        # Use H*H = I but make sure it can only happen if no gate is 
        # present in between 
        i = 0
        while G[i] != "MEASURE":
          if G[i] == "H":
            flag = 0
            #print(G[i],TH[i],AC1[i],AC2[i],"before start")
            j = i+1
            while G[j] != "MEASURE":
               if ((G[j] == "CZ" and AC1[j] == AC1[i]) or (G[j] == "CZ" and AC2[j] == AC1[i]) or (G[j] == "RZ" and AC1[j] == AC1[i])) :
                  break
               if G[j] == G[i] and AC1[j] == AC1[i] :
                  #print(G[i],TH[i],AC1[i],AC2[i],"before")
                  del G[j]
                  TH = np.delete(TH,j)
                  AC1 = np.delete(AC1,j)
                  AC2 = np.delete(AC2,j)
                  #print(G[i],TH[i],AC1[i],AC2[i],"after")
                  del G[i]
                  TH = np.delete(TH,i)
                  AC1 = np.delete(AC1,i)
                  AC2 = np.delete(AC2,i)
                  flag = 2
               j = j+1
               if flag ==2:
                  break 
          i = i + 1
        
        
        
        # Use CZ H RZ H CZ = RZ(pi/2) CZ RX(pi/2) RZ RX(-pi2) CZ RZ(-pi/2)
        i = 0 
        while G[i] != "MEASURE":
            if (G[i] == "CZ" and G[i+1] == "H" and AC2[i] == AC1[i+1] and G[i+2] == "RZ" and AC2[i] == AC1[i+2] and G[i+3] == "H" and AC2[i] == AC1[i+3] and G[i+4] == "CZ" and AC2[i] == AC2[i+4]):
                  G[i+1] = "RX"; TH[i+1] = 1.57079632679; 
                  G[i+3] = "RX"; TH[i+3] = -1.57079632679;
                  G.insert(i+5,"RZ"); TH = np.insert(TH,i+5,-1.57079632679); 
                  AC1 = np.insert(AC1,i+5,AC2[i]); AC2 = np.insert(AC2,i+5,0);
                  G.insert(i,"RZ"); TH = np.insert(TH,i,1.57079632679); 
                  AC1 = np.insert(AC1,i,AC2[i]); AC2 = np.insert(AC2,i,0);
                  print("loop activated")
            i = i+1
        
        
        # Use H = RZ(pi/2) RX(pi/2) RZ(pi/2)
        i = 0
        while G[i] !="MEASURE":
            if (G[i] == "H"):
                  flag = AC1[i]
                  G[i] = "RZ"; TH[i] = 1.57079632679 ;
                  G.insert(i,"RX");TH = np.insert(TH,i,1.57079632679);
                  AC1 = np.insert(AC1,i,flag); AC2 = np.insert(AC2,i,0); 
                  G.insert(i,"RZ");TH = np.insert(TH,i,1.57079632679); 
                  AC1 = np.insert(AC1,i,flag); AC2 = np.insert(AC2,i,0); 
            i = i+1 
        
        
        # Compress RZ gates
        loop_flag = 0
        for mm in range (0,1000):
         i = 0 
         while G[i] !="MEASURE": 
             if (G[i] == "RZ"):
                   j = i+1
                   flag = 0
                   #print(flag,"flag")
                   while G[j] !="MEASURE":
                         if (G[j] == "RX" and AC1[j] == AC1[i]):
                              flag = 2
                         if (G[j] == "RZ" and AC1[j] == AC1[i]):
                              TH[i] = TH[i]+TH[j]; 
                              del G[j];TH = np.delete(TH,j);
                              AC1 = np.delete(AC1,j); AC2 = np.delete(AC2,j) 
                              flag = 2
                              loop_flag = 3
                         j = j+1
                         if(flag == 2):
                              break
             if (G[i] == "RZ" and TH[i]== 0.0):
                   del G[i];TH = np.delete(TH,i);
                   AC1 = np.delete(AC1,i); AC2 = np.delete(AC2,i)
             i = i +1
         if(loop_flag == 0):
             break  
         if(mm ==1000 and loop_flag==3):
             print("more RZ compression are left be carefull!!")


        i = 0
        while G[i] != "MEASURE":
            if (G[i] == "RX" and TH[i] == 1.57079632679):
                i1 = i+1
                while G[i1] != "MEASURE":
                    if (G[i1] == "RX" and AC1[i1] == AC1[i] or G[i1] == "CZ" and AC1[i1] == AC1[i] or G[i1] == "RZ" and TH[i1] != 3.14159265358 and AC1[i1] == AC1[i] or G[i1] == "CZ" and AC2[i1] == AC1[i]):
                        break
                    if (G[i1] == "RZ" and TH[i1] == 3.14159265358 and AC1[i1] == AC1[i]):
                        i2 = i1+1
                        while G[i2] != "MEASURE":
                            if (G[i2] == "RX" and AC1[i2] == AC1[i] or G[i2] == "RZ" and AC1[i2] == AC1[i] ):
                                break
                            if (G[i2] == "CZ" and AC1[i2] == AC1[i] and G[i2+4] == "CZ" and AC1[i2+4] == AC1[i]):
                                i3 = i2 +5
                                while G[i3] != "MEASURE":
                                    if(G[i3] == "RZ" and AC1[i3] == AC1[i]+1 or G[i3] == "CZ" and AC1[i3] == AC1[i]+1 or G[i3] == "RX" and TH[i3] != 1.57079632679 and AC1[i3] == AC1[i]+1 ):
                                        break
                                    if(G[i3] == "RX" and TH[i3] == 1.57079632679 and AC1[i3] == AC1[i]+1) :
                                        i4 = i2 + 5
                                        while G[i4] != "MEASURE":
                                            if (G[i4] == "RZ" and AC1[i4] == AC1[i] or G[i4] == "CZ" and AC1[i4] == AC1[i] or G[i4] == "RX" and TH[i4] != 1.57079632679 and AC1[i4] == AC1[i] ):
                                                break
                                            if(G[i4] == "RX" and TH[i4] == 1.57079632679 and AC1[i4] == AC1[i]) :
                                                AC1[i2+1] = AC1[i];AC1[i2+2] = AC1[i];AC1[i2+3] = AC1[i]
                                                G[i4] = "RZ"; TH[i4] = 3.14159265358;
                                                del G[i3];TH = np.delete(TH,i3)
                                                AC1 = np.delete(AC1,i3); AC2 = np.delete(AC2,i3)
                                                G.insert(i2,"RX")
                                                TH = np.insert(TH,i2,1.57079632679)           
                                                AC1 = np.insert(AC1,i2,AC2[i2]); AC2 = np.insert(AC2,i2,0)
                                                del G[i1];TH = np.delete(TH,i1)
                                                AC1 = np.delete(AC1,i1); AC2 = np.delete(AC2,i1)
                                                del G[i];TH = np.delete(TH,i)
                                                AC1 = np.delete(AC1,i); AC2 = np.delete(AC2,i)
                                                break
                                            i4 = i4 +1
                                    i3 = i3 + 1
                            i2 = i2 + 1
                    i1 = i1 + 1 
            i = i+1
         
         
        # Compress RZ gates                                                            
        loop_flag = 0                                                                  
        for mm in range (0,1000):                                                      
            i = 0                                                                         
            while G[i] !="MEASURE":                                                       
                if (G[i] == "RZ"):                                                        
                    j = i+1                                                             
                    flag = 0                                                            
                    while G[j] !="MEASURE":                                             
                        if (G[j] == "RX" and AC1[j] == AC1[i]):                       
                            flag = 2                                                 
                        if (G[j] == "RZ" and AC1[j] == AC1[i]):                       
                            TH[i] = TH[i]+TH[j];                                     
                            del G[j];TH = np.delete(TH,j);                           
                            AC1 = np.delete(AC1,j); AC2 = np.delete(AC2,j)           
                            flag = 2                                                 
                            loop_flag = 3                                            
                        j = j+1                                                       
                        if(flag == 2):                                                
                            break                                                    
                if (G[i] == "RZ" and TH[i]== 0.0):                                        
                    del G[i];TH = np.delete(TH,i);                                      
                    AC1 = np.delete(AC1,i); AC2 = np.delete(AC2,i)                      
                i = i +1                                                                  
            if(loop_flag == 0):                                                           
                break                                                                     
            if(mm ==1000 and loop_flag==3):                                               
                print("more RZ compression are left be carefull!!")
         
         
        # Use RZ(theta) RX RZ(pi) = RZ(theta-pi) RX(-pi2)
        i = 0
        while G[i] != "MEASURE":
            if (G[i] == "RZ"):
                loop_breaker = 0
                i1 = i+1
                while G[i1] != "MEASURE":
                    if (G[i1] == "RX" and TH[i1] != 1.57079632679 and AC1[i1] == AC1[i]):
                        break
                    if (G[i1] == "RX" and TH[i1] == 1.57079632679 and AC1[i1] == AC1[i]):
                        i2 = i1 + 1
                        while G[i2] != "MEASURE":
                            if (G[i2] == "RZ" and TH[i2] == 3.14159265358 and AC1[i2] == AC1[i]):
                                TH[i] = TH[i]+TH[i2]
                                TH[i1] = -TH[i1]
                                del G[i2];TH = np.delete(TH,i2);
                                AC1 = np.delete(AC1,i2); AC2 = np.delete(AC2,i2);
                                loop_breaker = 3
                                break
                            elif (G[i2] == "RZ" and TH[i2] != 3.14159265358 and AC1[i2] == AC1[i]):
                                loop_breaker = 3
                                break
                            i2 = i2 + 1                         
                    if (loop_breaker == 3):
                        break
                    i1 = i1 + 1
            i = i + 1

        
        p = Program(RESET()) #compressed program
        ro = p.declare('ro', memory_type='BIT', memory_size=nqubits)

        for i in range(len(G)):
            if (G[i] == "RX"):
                p.inst(RX(TH[i], int(AC1[i])))
            if (G[i] == "RZ"):
                p.inst(RZ(TH[i], int(AC1[i])))
            if (G[i] == "CZ"):
                p.inst(CZ(int(AC1[i]), int(AC2[i])))
        for i in range(0,nqubits):
            p.inst(MEASURE(i, ro[i]))
        return p



