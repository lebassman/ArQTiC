#import necessary libraries
import numpy as np
from arqtic.program import Program, Gate
#from arqtic.qite import make_QITE_program
from arqtic.arqtic_for_ibm import ibm_circ_to_program, get_ibm_circuit
#from arqtic.ds_compiler import get_constant_depth_program
from arqtic.real_time import heisenberg_evolution_program, heisenberg2D_evolution_program
from arqtic.observables import *
from arqtic.exceptions import Error
import os

#Create data directory
current=os.getcwd()
newdir="data"
path = os.path.join(current, newdir) 
if not os.path.isdir(path):
    os.makedirs(path)



class Simulation_Generator:

    def __init__(self,file="",log="logfile.txt"):
        #%matplotlib inline
        completename = os.path.join(path,log)
        self.namevar=str(completename)
        with open(self.namevar,'w') as tempfile:
            tempfile.write("***ArQTiC Session Log File***\n\n")

        #Default Parameters
        self.H_BAR = 1 #value of h-bar
        self.Jx=self.Jy=self.Jz=[] #arrays for coupling constants
        self.hx=self.hy=self.hz=[] #arrays for external fields
        self.td_Jx_func=self.td_Jy_func=self.td_Jz_func=[] #time-dependence functions for coupling parameters
        self.td_hx_func=self.td_hy_func=self.td_hz_func=[] #time-dependence functions for external fields
        self.num_spins=[2] #total number of qubits along each lattice dimension
        self.dims = 1
        self.nRows = 1
        self.nCols = 1
        self.nLayers = 1
        self.initial_spins=[] #only for setting initial product states
        self.delta_t=1
        self.steps=1
        self.start_timestep=0
        self.end_timestep=0
        self.real_time="True"
        self.QCQS="QS"
        self.shots=1024
        self.noise_choice="False"
        self.device="ibmq_qasm_simulator"
        self.plot_flag="True"
        self.freq=0
        self.time_dep_flag="False"
        self.custom_time_dep="False"
        self.programs_list=[]
        self.backend="ibm"
        self.ibm_circuits_list=[]
        self.rigetti_circuits_list=[]
        self.cirq_circuits_list=[]
        self.qite_energies = []
        self.compile="False"
        self.compiler="native"
        self.constant_depth="False"
        self.observable="system_magnetization"
        self.measure_dir="z"
        self.PBC = "False"

        self.time_func=np.cos
        
        #if input file was given, read in values from input file
        if (file != ""):
            input_file=open(file,'r')
            data=input_file.readlines()
            for i in range(len(data)-1):
                value=data[i+1].strip()
                if "*Jx" in data[i]:
                    self.Jx=value.split(' ')
                elif "*Jy" in data[i]:
                    self.Jy=value.split(' ')
                elif "*Jz" in data[i]:
                    self.Jz=value.split(' ')
                elif "*hx" in data[i]:
                    self.hx=value.split(' ')
                elif "*hy" in data[i]:
                    self.hy=value.split(' ')
                elif "*hz" in data[i]:
                    self.hz=value.split(' ')
                elif "*td_Jx_func" in data[i]:
                    self.td_Jx_func=value.split(' ')
                elif "*td_Jy_func" in data[i]:
                    self.td_Jy_func=value.split(' ')
                elif "*td_Jz_func" in data[i]:
                    self.td_Jz_func=value.split(' ')
                elif "*td_hx_func" in data[i]:
                    self.td_hx_func=value.split(' ')
                elif "*td_hy_func" in data[i]:
                    self.td_hy_func=value.split(' ')
                elif "*td_hz_func" in data[i]:
                    self.td_hz_func=value.split(' ')
                elif "*hbar" in data[i]:
                    if (value == "eVfs"):
                        self.H_BAR = 0.658212  # eV*fs
                elif "*initial_spins" in data[i]:
                    self.initial_spins=value.split(' ')
                elif "*delta_t" in data[i]:
                    self.delta_t=float(value)
                elif "*steps" in data[i]:
                    self.steps=int(value)
                elif "*start_timestep" in data[i]:
                    self.start_timestep=int(value)
                elif "*end_timestep" in data[i]:
                    self.end_timestep=int(value)
                elif "*real_time" in data[i]:
                    self.real_time=value
                elif "*observable" in data[i]:
                    self.observable=value
                elif "*beta" in data[i]:
                    self.beta=float(value)
                elif "*delta_beta" in data[i]:
                    self.delta_beta=float(value)
                elif "*domain" in data[i]:
                    self.domain=int(value)
                elif "*num_spins" in data[i]:
                    self.num_spins = 1
                    dims = value.split(' ')
                    self.dims = len(dims)
                    if (len(dims) > 0):
                        self.nRows = int(dims[0])
                        self.num_spins*=self.nRows
                    if (len(dims) > 1):
                        self.nCols = int(dims[1])
                        self.num_spins*=self.nCols
                    if (len(dims) > 2):
                        self.nLayers = int(dims[2])
                        self.num_spins*=self.nLayers
                elif "*QCQS" in data[i]:
                    self.QCQS=value
                elif "*device" in data[i]:
                    self.device=value
                elif "*backend" in data[i]:
                    self.backend=value
                elif "*measure_dir" in data[i]:
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
                elif "*compiler" in data[i]:
                    self.compiler=value
                elif "*compile" in data[i]:
                    self.compile=value
                elif "*constant_depth" in data[i]:
                    self.constant_depth=value
                elif "*PBC" in data[i]:
                    self.PBC=value
                elif "*custom_time_dep" in data[i]:
                    self.custom_time_dep=value
                    if self.custom_time_dep in "True":
                        from time_dependence import external_func
                        print("Found an external time dependence function")
                        with open(self.namevar,'a') as tempfile:
                            tempfile.write("Found an external time dependence function\n")
                        self.time_func=external_func
           
        #format array entries
        #initial spin state
        if (self.initial_spins == []):
            self.initial_spins = np.zeros(self.num_spins)
        else:
            self.initial_spins = np.asarray(self.initial_spins)
        

    def generate_programs(self):
        programs = []
        #create programs for real_time evolution
        if (self.real_time == "True"):
            if (self.end_timestep == 0):
                self.end_timestep = self.steps+1
            #inital spin preparation program
            init_prog = Program(self.num_spins)
            index=0
            for q in self.initial_spins:
                if int(q)==1:
                    init_prog.add_gate(Gate([index], 'X'))
                    index+=1
                else: 
                    index+=1
                                     
            #measurement preparation program
            meas_prog = Program(self.num_spins)
            if "x" in self.measure_dir:
                for q in range(self.num_spins):
                    meas_prog.add_gate(Gate([q],'H',))
            elif "y" in self.measure_dir:
                for q in range(self.num_spins):
                    meas_prog.add_gate(Gate([q],'RX',angles=[-np.pi/2]))
                                     
            #total program
            for j in range(self.start_timestep, self.end_timestep+1):
                print("Generating timestep {} program".format(j))
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Generating timestep {} program\n".format(j))
                evolution_time = self.delta_t * j
                if (self.dims == 1):
                    evol_prog = heisenberg_evolution_program(self, evolution_time)
                elif (self.dims == 2): 
                    evol_prog = heisenberg2D_evolution_program(self, evolution_time)
                else: 
                    raise Error(f"System dimension of size {self.dim} not yet implemented.")
                if (self.constant_depth == "True" and j>0):
                    evol_prog = get_constant_depth_program(evol_prog, self.num_spins)
                total_prog = Program(self.num_spins)
                total_prog.append_program(init_prog)
                total_prog.append_program(evol_prog)
                total_prog.append_program(meas_prog)
                programs.append(total_prog)
            self.programs_list=programs
  
        #create programs for imaginary time evolution
        else:
            qite_program, energies = make_QITE_program(self)
            programs.append(qite_program)
            self.qite_energies = energies
            self.programs_list=programs

    def generate_circuits(self):
        if (self.programs_list == []):
            self.generate_programs()
        #convert to backend specific circuits if one is requested    
        if self.backend in "ibm":
            self.generate_ibm()
        if self.backend in "rigetti":
            self.generate_rigetti()
        if self.backend in "cirq":
            self.generate_cirq()

    def generate_ibm(self):
        #generate IBM-specific circuits
        #IBM imports 
        import qiskit as qk
        from qiskit.tools.monitor import job_monitor
        from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
        from qiskit import Aer, IBMQ, execute
        from qiskit.providers.aer import noise
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.circuit import quantumcircuit
        from qiskit.circuit import Instruction
        

        print("Creating IBM quantum circuit objects...")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Creating IBM quantum circuit objects...\n")
        name=0
        self.q_regs = qk.QuantumRegister(self.num_spins, 'q')
        self.c_regs = qk.ClassicalRegister(self.num_spins, 'c')
        backend = self.device
        self.ibm_circuits_list=[]
        for program in self.programs_list:
            ibm_circ = get_ibm_circuit(backend, program,self.q_regs,self.c_regs,self.device)
            self.ibm_circuits_list.append(ibm_circ)
        print("IBM quantum circuit objects created")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("IBM quantum circuit objects created\n")

        if (self.compile == "True"):
            provider = qk.IBMQ.get_provider(group='open')
            #print(provider.backends())
            device = provider.get_backend(self.device)
            if (self.compiler == "native"):
                print("Transpiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Transpiling circuits...\n")
                temp=qk.compiler.transpile(self.ibm_circuits_list,backend=device,optimization_level=2)
                self.ibm_circuits_list=temp
                print("Circuits transpiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits transpiled successfully\n")
            elif (self.compiler == "tket"):
                from pytket.qiskit import qiskit_to_tk
                from pytket.backends.ibm import IBMQBackend, IBMQEmulatorBackend, AerBackend
                from pytket.qasm import circuit_to_qasm_str
                if self.device == "":
                    tket_backend = AerBackend()
                else: 
                    if (self.QCQS == "QC"):
                        tket_backend = IBMQBackend(self.device)
                    else: 
                        tket_backend = IBMQEmulatorBackend(self.device)
                print("Compiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Compiling circuits...\n")
                circs = []
                for circuit in self.ibm_circuit_list:
                    tket_circ = qiskit_to_tk(circuit)
                    tket_backend.compile_circuit(tket_circ)
                    qasm_str = circuit_to_qasm_str(tket_circ)
                    ibm_circ = qk.QuantumCircuit.from_qasm_str(qasm_str)
                    circs.append(tket_circ)
                self.ibm_circuits_list=circs
                print("Circuits compiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits compiled successfully\n")
             


    def generate_rigetti(self):
        import pyquil
        from pyquil.quil import Program
        from pyquil.gates import H, RX, RZ, CZ, RESET, MEASURE
        from pyquil.api import get_qc
        self.rigetti_circuits_list=[]
        #Rigettti imports
        import pyquil
        from pyquil.quil import Program
        from pyquil.gates import H, RX, RZ, CZ, RESET, MEASURE
        from pyquil.api import get_qc

        print("Creating Pyquil program list...")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Creating Pyquil program list...\n")
        for circuit in self.programs_list:
            p = pyquil.Program(RESET()) #compressed program
            ro = p.declare('ro', memory_type='BIT', memory_size=self.num_spins)
            for gate in circuit.gates:
                if gate.name != "":
                    if gate.name in "X":
                        p.inst(pyquil.gates.X(gate.qubits[0]))
                    elif gate.name in "Y":
                        p.inst(pyquil.gates.Y(gate.qubits[0]))
                    elif gate.name in "Z":
                        p.inst(pyquil.gates.Z(gate.qubits[0]))
                    elif gate.name in "H":
                        p.inst(pyquil.gates.H(gate.qubits[0]))
                    elif gate.name in "RZ":
                        p.inst(pyquil.gates.RZ(gate.angles[0],gate.qubits[0]))
                    elif gate.name in "RX":
                        p.inst(pyquil.gates.RX(gate.angles[0],gate.qubits[0]))
                    elif gate.name in "CNOT":
                        p.inst(pyquil.gates.CNOT(gate.qubits[0],gate.qubits[1]))
                    else:
                        print("Unrecognized gate: {}".format(gate.name))
            for i in range(self.num_spins):
                p.inst(pyquil.gates.MEASURE(i,ro[i]))
            p.wrap_in_numshots_loop(self.shots)
            self.rigetti_circuits_list.append(p)

        if "True" in self.compile:
            if self.QCQS in ["QS"]:
                qc=get_qc(self.device, as_qvm=True)
            else:
                qc=get_qc(self.device)
            qc.compiler.set_timeout(100)

            if self.compiler in "native":
                temp=[]
                print("Transpiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Transpiling circuits...\n")
                for circuit in self.rigetti_circuits_list:
                    circ = qc.compile(circuit)
                    temp.append(circ)
                self.rigetti_circuits_list=temp
                print("Circuits transpiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits transpiled successfully\n")
            elif self.compiler in "tket":
                temp=[]
                from pytket.pyquil import pyquil_to_tk
                from pytket.backends.forest import ForestBackend
                if self.device == "":
                    qvm = '{}q-qvm'.format(self.num_spins)
                    tket_backend = ForestBackend(qvm, simulator=True)
                else:
                    if self.QCQS in ["QC"]:
                        tket_backend = ForestBackend(self.device)
                    else:
                        tket_backend = ForestBackend(self.device, simulator=True)
                print("Compiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Compiling circuits...\n")
                for circuit in self.rigetti_circuits_list:
                    tket_circ = qiskit_to_tk(circuit)
                    tket_backend.compile_circuit(tket_circ)
                    temp.append(tket_circ)
                self.ibm_circuits_list=temp
                print("Circuits compiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits compiled successfully\n")


        print("Pyquil program list created successfully")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Pyquil program list created successfully\n")

    def generate_cirq(self):
        self.cirq_circuits_list=[]
        #Cirq imports
        import cirq
        print("Creating Cirq circuit list...")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Creating Cirq circuit list...\n")
        for circuit in self.programs_list:
            c=cirq.Circuit()
            qubit_list=cirq.LineQubit.range(self.num_spins)
            gate_list=[]
            for gate in circuit.gates:
                if gate.name in "H":
                    gate_list.append(cirq.H(qubit_list[gate.qubits[0]]))
                elif gate.name in "S":
                    gate_list.append(cirq.Z(qubit_list[gate.qubits[0]])**(0.5))
                elif gate.name in "SDG":
                    gate_list.append(cirq.Z(qubit_list[gate.qubits[0]])**(-0.5))
                elif gate.name in "SX":
                    gate_list.append(cirq.X(qubit_list[gate.qubits[0]])**(0.5))
                elif gate.name in "SXDG":
                    gate_list.append(cirq.X(qubit_list[gate.qubits[0]])**(-0.5))
                elif gate.name in "RZ":
                    gate_list.append(cirq.rz(gate.angles[0])(qubit_list[gate.qubits[0]]))
                elif gate.name in "RX":
                    gate_list.append(cirq.rx(gate.angles[0])(qubit_list[gate.qubits[0]]))
                elif gate.name in "CNOT":
                    gate_list.append(cirq.CNOT(qubit_list[gate.qubits[0]],qubit_list[gate.qubits[1]]))
                elif gate.name in "ISwapPowGate":
                    gate_list.append(cirq.ISwapPowGate(exponent=(-1.*gate.angles[0]/np.pi)).on(qubit_list[gate.qubits[0]],qubit_list[gate.qubits[1]]))
            #for i in range(self.num_spins):
            #    gate_list.append(cirq.measure(qubit_list[i]))
            c.append(gate_list,strategy=cirq.InsertStrategy.EARLIEST)
            self.cirq_circuits_list.append(c)
        print("Successfully created Cirq circuit list")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Successfully created Cirq circuit list\n")


    def connect_IBM(self,api_key=None, overwrite=False):
        import qiskit as qk
        if api_key != None:
            if overwrite==False:
                qk.IBMQ.save_account(api_key) ## only run once!
            else:
                qk.IBMQ.save_account(api_key,overwrite=True) ## only run once!
     #qk.IBMQ.delete_accounts() ## only run if you need to use a new token
        qk.IBMQ.load_account()


    #def parameters(self):
        #print("Current model parameters:\n\nH_BAR = {}\nJX = {}\nJY = {}\nJZ = {}\nh_ext = {}\next_dir = {}".format(self.H_BAR,self.JX,self.JY,self.JZ,self.h_ext,self.ext_dir))
        #print("num_spins = {}\ninitial_spins = {}\ndelta_t = {}\nsteps = {}\nQCQS = {}\nshots = {}\nnoise_choice = {}".format(self.num_spins,self.initial_spins,self.delta_t,self.steps,self.QCQS,self.shots,self.noise_choice))
        #print("device choice = {}\nplot_flag = {}\nfreq = {}\ntime_dep_flag = {}\ncustom_time_dep = {}\n".format(self.device,self.plot_flag,self.freq,self.time_dep_flag,self.custom_time_dep)) 
        #print("compile = {}\nauto_smart_compile = {}\ndefault_compiler = {}".format(self.compile,self.auto_smart_compile,self.compiler))
        #this is missing some of the latest parameter additions

    def results(self):
        return self.result_matrix

    def return_circuits(self):
        if self.backend in "ibm":
            if len(self.ibm_circuits_list)==0:
                self.generate_circuits()
            return self.ibm_circuits_list
        elif self.backend in "rigetti":
            if len(self.rigetti_circuits_list)==0:
                self.generate_circuits()
            return self.rigetti_circuits_list
        elif self.backend in "cirq":
            if len(self.cirq_circuits_list)==0:
                self.generate_circuits()
            return self.cirq_circuits_list
        
    def write_programs_to_qasm(self, fname="my_circuit"):
        if (self.programs_list == []):
            self.generate_programs()
        for t in range(self.end_timestep+1 - self.start_timestep):
            timestep = self.start_timestep + t
            file = open(f'{fname}_timestep{timestep}.qasm', 'w')
            file.write("""OPENQASM 2.0;\ninclude "qelib1.inc";\n\n""")
            file.write(f"qreg q[{self.num_spins}];\n\n")
            program = self.programs_list[t]
            for gate in program.gates:
                if gate.name in "H":
                    file.write(f"h q[{gate.qubits[0]}];\n")
                elif gate.name in "X":
                    file.write(f"x q[{gate.qubits[0]}];\n")
                elif gate.name in "Y":
                    file.write(f"y q[{gate.qubits[0]}];\n")
                elif gate.name in "Z":
                    file.write(f"z q[{gate.qubits[0]}];\n")
                elif gate.name in "S":
                    file.write(f"s q[{gate.qubits[0]}];\n")
                elif gate.name in "SDG":
                    file.write(f"sdg q[{gate.qubits[0]}];\n")
                elif gate.name in "SX":
                    file.write(f"sx q[{gate.qubits[0]}];\n")
                elif gate.name in "SXDG":
                    file.write(f"sxdg q[{gate.qubits[0]}];\n")
                elif gate.name in "RX":
                    file.write(f"rx({gate.angles[0]}) q[{gate.qubits[0]}];\n")
                elif gate.name in "RY":
                    file.write(f"ry({gate.angles[0]}) q[{gate.qubits[0]}];\n")
                elif gate.name in "RZ":
                    file.write(f"rz({gate.angles[0]}) q[{gate.qubits[0]}];\n")
                elif gate.name in "CNOT":
                    file.write(f"cx q[{gate.qubits[0]}], q[{gate.qubits[1]}];\n")
                else: 
                    raise Error(f"gate name {gate.name} not recognized!")
            file.close()


    def run_circuits(self):
        import glob
        if "True" in self.plot_flag:
            import matplotlib.pyplot as plt
        if self.backend in "ibm":
            import qiskit as qk
            from qiskit.tools.monitor import job_monitor
            from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
            from qiskit import Aer, IBMQ, execute
            from qiskit.providers.aer import noise
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit.circuit import quantumcircuit
            from qiskit.circuit import Instruction
            
            q_regs = qk.QuantumRegister(self.num_spins, 'q')
            c_regs = qk.ClassicalRegister(self.num_spins, 'c')
            ## Show available backends
            provider = qk.IBMQ.get_provider(group='open')
            provider.backends()

            #choose the device you would like to run on
            device = provider.get_backend(self.device)
            #gather fidelity statistics on this device if you want to create a noise model for the simulator
            if (self.device != "ibmq_qasm_simulator"):
                properties = device.properties()
                coupling_map = device.configuration().coupling_map
                noise_model = NoiseModel.from_backend(device)
                basis_gates = noise_model.basis_gates
            
            #add measurements
            temp = []
            for circ in self.ibm_circuits_list:
                circ.measure(self.q_regs,self.c_regs)
                temp.append(circ)
            self.ibm_circuits_list = temp
            #compile circuits to run
            temp = qk.compiler.transpile(self.ibm_circuits_list,backend=device,optimization_level=2)
            self.ibm_circuits_list = temp

            #CHOOSE TO RUN ON QUANTUM COMPUTER OR SIMULATOR
            if self.QCQS in ["QC"]:
                #quantum computer execution
                job = qk.execute(self.ibm_circuits_list, backend=device, shots=self.shots)
                job_monitor(job)

            elif self.QCQS in ["QS"]:
                simulator = Aer.get_backend('qasm_simulator')
                #simulator execution
                if self.noise_choice in ["True"]:
                    print("Running noisy simulator job...")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Running noisy simulator job...\n")
                    result_noise = execute(self.ibm_circuits_list, simulator, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates,shots=self.shots).result()
                    print("Noisy simulator job successful")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Noisy simulator job successful\n")
                elif self.noise_choice in ["False"]:
                    print("Running noiseless simulator job...")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Running noiseless simulator job...\n")
                        result_noise=execute(self.ibm_circuits_list,backend=simulator,shots=self.shots).result()
                    print("Noiseless simulator job successful")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Noiseless simulator job successful\n")
                else: 
                    print("Please enter either True or False for the simulator noise query")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Please enter either y or n for the simulator noise query\n")
            else:
                print("Please enter either QC or QS")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Please enter either QC or QS\n")


            #Post Processing Depending on Choice
            self.result_out_list=[]
            if self.QCQS in ["QS"]:
                #SIMULATOR POST PROCESSING
                if (self.observable == "local_magnetization"):
                    for j in range(self.num_spins):
                        avg_mag_sim = []
                        temp = []
                        i = 1
                        print("Post-processing qubit {} data".format(j+1))
                        with open(self.namevar,'a') as tempfile:
                            tempfile.write("Post-processing qubit {} data\n".format(j+1))
                        for c in self.ibm_circuits_list:
                            result_dict = result_noise.get_counts(c)
                            temp.append(local_magnetization(self.num_spins,result_dict, self.shots,j))
                            if i % (self.steps+1) == 0:
                                avg_mag_sim.append(temp)
                                temp = []
                            i += 1
                        # time_vec=np.linspace(0,total_t,steps)
                        # time_vec=time_vec*JX/H_BAR
                        if "True" in self.plot_flag:
                            fig, ax = plt.subplots()
                            plt.plot(range(self.steps+1), avg_mag_sim[0])
                            plt.xlabel("Simulation Timestep",fontsize=14)
                            plt.ylabel("Average Magnetization",fontsize=14)
                            plt.tight_layout()
                            every_nth = 2
                            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                                if (n+1) % every_nth != 0:
                                    label.set_visible(False)
                            every_nth = 2
                            for n, label in enumerate(ax.yaxis.get_ticklabels()):
                                if (n+1) % every_nth != 0:
                                    label.set_visible(False)
                            # plt.yticks(np.arange(-1, 1, step=0.2))  # Set label locations.
                            plt.savefig("data/Simulator_result_qubit{}.png".format(j+1))
                            plt.close()
                        self.result_out_list.append(avg_mag_sim[0])
                        existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, self.num_spins))
                        np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,self.num_spins,len(existing)+1),avg_mag_sim[0])
                    self.result_matrix=np.stack(self.result_out_list)
                    print("Done")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Done\n")
                elif (self.observable == "staggered_magnetization"):
                    avg_sm = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        avg_sm.append(staggered_magnetization(result_dict, self.shots))
                    #plt.plot.range(self.steps+1), avg_zs[0])
                    #plt.xlabel("Simulation Timestep",fontsize=14)
                    #plt.ylabel("Order Parameter",fontsize=14)  
                    #plt.savefig("data/order_param.png")
                    #plt.close()
                    self.result_out_list.append(avg_sm)
                    self.result_matrix=avg_sm
                elif (self.observable == "system_magnetization"):
                    avg_mag = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        avg_mag.append(system_magnetization(result_dict, self.shots))
                    #plt.plot.range(self.steps+1), avg_zs[0])
                    #plt.xlabel("Simulation Timestep",fontsize=14)
                    #plt.ylabel("Order Parameter",fontsize=14)  
                    #plt.savefig("data/order_param.png")
                    #plt.close()
                    self.result_out_list.append(avg_mag)
                    self.result_matrix=avg_mag
                elif (self.observable == "excitation_displacement"):
                    disp = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        disp.append(excitation_displacement(self.num_spins,result_dict, self.shots))
                    self.result_out_list.append(disp)
                    self.result_matrix=disp

            elif self.QCQS in ["QC"]:
                #QUANTUM COMPUTER POST PROCESSING
                result_noise=job.result()
                if (self.observable == "local_magnetization"):
                    for j in range(self.num_spins):
                        avg_mag_sim = []
                        temp = []
                        i = 1
                        print("Post-processing qubit {} data".format(j+1))
                        with open(self.namevar,'a') as tempfile:
                            tempfile.write("Post-processing qubit {} data\n".format(j+1))
                        for c in self.ibm_circuits_list:
                            result_dict = result_noise.get_counts(c)
                            temp.append(local_magnetization(self.num_spins, result_dict, self.shots,j))
                            if i % (self.steps+1) == 0:
                                avg_mag_sim.append(temp)
                                temp = []
                            i += 1
                        # time_vec=np.linspace(0,total_t,steps)
                        # time_vec=time_vec*JX/H_BAR
                        if "True" in self.plot_flag:
                            fig, ax = plt.subplots()
                            plt.plot(range(self.steps+1), avg_mag_sim[0])
                            plt.xlabel("Simulation Timestep",fontsize=14)
                            plt.ylabel("Average Magnetization",fontsize=14)
                            plt.tight_layout()
                            every_nth = 2
                            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                                if (n+1) % every_nth != 0:
                                    label.set_visible(False)
                            every_nth = 2
                            for n, label in enumerate(ax.yaxis.get_ticklabels()):
                                if (n+1) % every_nth != 0:
                                    label.set_visible(False)
                            # plt.yticks(np.arange(-1, 1, step=0.2))  # Set label locations.
                            plt.savefig("data/Simulator_result_qubit{}.png".format(j+1))
                            plt.close()
                        self.result_out_list.append(avg_mag_sim[0])
                        existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, self.num_spins))
                        np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,self.num_spins,len(existing)+1),avg_mag_sim[0])
                    self.result_matrix=np.stack(self.result_out_list)
                    print("Done")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Done\n")
                elif (self.observable == "staggered_magnetization"):
                    avg_sm = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        avg_sm.append(staggered_magnetization(result_dict, self.shots))
                    #plt.plot.range(self.steps+1), avg_zs[0])
                    #plt.xlabel("Simulation Timestep",fontsize=14)
                    #plt.ylabel("Order Parameter",fontsize=14)  
                    #plt.savefig("data/order_param.png")
                    #plt.close()
                    self.result_out_list.append(avg_sm)
                    self.result_matrix=avg_sm
                elif (self.observable == "system_magnetization"):
                    avg_mag = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        avg_mag.append(system_magnetization(result_dict, self.shots))
                    #plt.plot.range(self.steps+1), avg_zs[0])
                    #plt.xlabel("Simulation Timestep",fontsize=14)
                    #plt.ylabel("Order Parameter",fontsize=14)  
                    #plt.savefig("data/order_param.png")
                    #plt.close()
                    self.result_out_list.append(avg_mag)
                    self.result_matrix=avg_mag
                elif (self.observable == "excitation_displacement"):
                    disp = []
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        disp.append(excitation_displacement(self.num_spins, result_dict, self.shots))
                    self.result_out_list.append(disp)
                    self.result_matrix=disp
        elif "rigetti" in self.backend:
            import pyquil
            from pyquil.quil import Program
            from pyquil.gates import H, RX, RZ, CZ, RESET, MEASURE
            from pyquil.api import get_qc
            print("Running Pyquil programs...")
            with open(self.namevar,'a') as tempfile:
                tempfile.write("Running Pyquil programs...\n")
            qc=get_qc(self.device)
            results_list=[]
            first_ind=0
            #each circuit represents one timestep
            for circuit in self.rigetti_circuits_list:
                temp=qc.run(circuit)
                results_list.append(temp)


            #Post Processing Depending on Choice
            self.result_out_list=[]
            #SIMULATOR POST PROCESSING
            if (self.observable == "local_magnetization"):
                for j in range(self.num_spins):
                    avg_mag_sim = []
                    temp = []
                    i = 1
                    print("Post-processing qubit {} data".format(j+1))
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Post-processing qubit {} data\n".format(j+1))
                    for result in results_list:
                        temp.append(local_magnetization_rigetti(self.num_spins,result, self.shots,j))
                        if i % (self.steps+1) == 0:
                            avg_mag_sim.append(temp)
                            temp = []
                        i += 1
                    # time_vec=np.linspace(0,total_t,steps)
                    # time_vec=time_vec*JX/H_BAR
                    if "True" in self.plot_flag:
                        fig, ax = plt.subplots()
                        plt.plot(range(self.steps+1), avg_mag_sim[0])
                        plt.xlabel("Simulation Timestep",fontsize=14)
                        plt.ylabel("Average Magnetization",fontsize=14)
                        plt.tight_layout()
                        every_nth = 2
                        for n, label in enumerate(ax.xaxis.get_ticklabels()):
                            if (n+1) % every_nth != 0:
                                label.set_visible(False)
                        every_nth = 2
                        for n, label in enumerate(ax.yaxis.get_ticklabels()):
                            if (n+1) % every_nth != 0:
                                label.set_visible(False)
                        # plt.yticks(np.arange(-1, 1, step=0.2))  # Set label locations.
                        plt.savefig("data/Simulator_result_qubit{}.png".format(j+1))
                        plt.close()
                    self.result_out_list.append(avg_mag_sim[0])
                    existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, self.num_spins))
                    np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,self.num_spins,len(existing)+1),avg_mag_sim[0])
                self.result_matrix=np.stack(self.result_out_list)
                print("Done")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Done\n")
            elif (self.observable == "staggered_magnetization"):
                avg_sm = []
                for result in results_list:
                    avg_sm.append(staggered_magnetization_rigetti(result, self.shots))
                #plt.plot.range(self.steps+1), avg_zs[0])
                #plt.xlabel("Simulation Timestep",fontsize=14)
                #plt.ylabel("Order Parameter",fontsize=14)  
                #plt.savefig("data/order_param.png")
                #plt.close()
                self.result_out_list.append(avg_sm)
                self.result_matrix=avg_sm
            elif (self.observable == "system_magnetization"):
                avg_mag = []
                for result in results_list:
                    avg_mag.append(system_magnetization_rigetti(result, self.shots))
                #plt.plot.range(self.steps+1), avg_zs[0])
                #plt.xlabel("Simulation Timestep",fontsize=14)
                #plt.ylabel("Order Parameter",fontsize=14)  
                #plt.savefig("data/order_param.png")
                #plt.close()
                self.result_out_list.append(avg_mag)
                self.result_matrix=avg_mag
            elif (self.observable == "excitation_displacement"):
                disp = []
                for result in results_list:
                    disp.append(excitation_displacement_rigetti(self.num_spins,result, self.shots))
                self.result_out_list.append(disp)
                self.result_matrix=disp

            
