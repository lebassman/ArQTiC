#import necessary libraries
import numpy as np
from arqtic.program import Program, Gate
from arqtic.ds_compiler import ds_compile
import os

#Create data directory
current=os.getcwd()
newdir="data"
path = os.path.join(current, newdir) 
if not os.path.isdir(path):
    os.makedirs(path)



class Simulation_Generator:

    def __init__(self,file="input_file.txt",log="logfile.txt"):
        #%matplotlib inline 
        input_file=open(file,'r')
        data=input_file.readlines()
        completename = os.path.join(path,log)
        self.namevar=str(completename)
        with open(self.namevar,'w') as tempfile:
            tempfile.write("***ArQTiC Session Log File***\n\n")
        #self.H_BAR = 0.658212    # eV*fs
        self.H_BAR = 1

        #Default Parameters
        self.Jx=self.Jy=self.Jz=self.h_ext=0
        self.ext_dir="Z"
        self.num_spins=2
        self.initial_spins=[]
        self.delta_t=1
        self.steps=1
        self.real_time="True"
        self.QCQS="QS"
        self.shots=1024
        self.noise_choice="False"
        self.device=""
        self.plot_flag="True"
        self.freq=0
        self.time_dep_flag="False"
        self.custom_time_dep="False"
        self.programs_list=[]
        self.backend=""
        self.ibm_circuits_list=[]
        self.rigetti_circuits_list=[]
        self.cirq_circuits_list=[]
        self.compile="False"
        self.compiler="native"
        self.observable="system_magnetization"
        self.observable_dir=["z"]

        from numpy import cos as cos_func
        self.time_func=cos_func

        for i in range(len(data)-1):
            value=data[i+1].strip()
            if "*Jx" in data[i]:
                self.Jx=float(value)
            elif "*Jy" in data[i]:
                self.Jy=float(value)
            elif "*Jz" in data[i]:
                self.Jz=float(value)
            elif "*h_ext" in data[i]:
                self.h_ext=float(value)
            elif "*ext_dir" in data[i]:
                self.ext_dir=value
            elif "*initial_spins" in data[i]:
                self.initial_spins=value
            elif "*delta_t" in data[i]:
                self.delta_t=float(value)
            elif "*steps" in data[i]:
                self.steps=int(value)
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
                self.num_spins=int(value)
            elif "*QCQS" in data[i]:
                self.QCQS=value
            elif "*device" in data[i]:
                self.device=value
            elif "*backend" in data[i]:
                self.backend=value
            elif "*observable_dir" in data[i]:
                self.observable_dir=[value]
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
            elif "*custom_time_dep" in data[i]:
                self.custom_time_dep=value
                if self.custom_time_dep in "True":
                    from time_dependence import external_func
                    print("Found an external time dependence function")
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Found an external time dependence function\n")
                    self.time_func=external_func

        self.total_time=int(self.delta_t*self.steps)
        if (self.initial_spins == []):
            self.initial_spins=np.zeros(self.num_spins)


        if "energy" in self.observable:
            self.observable_dir=[]
            if self.Jx != 0:
                self.observable_dir.append("x")
            if self.Jy != 0:
                self.observable_dir.append("y")
            if self.Jz != 0:
                self.observable_dir.append("z")
            if self.h_ext != 0:
                if "x" in self.ext_dir and not ("x" in self.observable_dir):
                    self.observable_dir.append("x")
                if "y" in self.ext_dir and not ("y" in self.observable_dir):
                    self.observable_dir.append("y")
                if "z" in self.ext_dir and not ("z" in self.observable_dir):
                    self.observable_dir.append("z")



    def heisenberg_evolution_program(self,evol_time): #creates evolution circuit in local program
    #Initial flipped spins are not implemented in this function due to the need for "barrier". Need to do that outside of this.
        prop_steps = int(evol_time / self.delta_t)  # number of propagation steps
        P=Program(self.num_spins)
        for i in range(len(self.observable_dir)):
            for step in range(prop_steps):
                t = (step + 0.5) * self.delta_t
                if "False" in self.time_dep_flag:
                    psi_ext = -2.0 * self.h_ext *self.delta_t / self.H_BAR
                elif "True" in self.time_dep_flag:
                    if "True" in self.custom_time_dep:
                        psi_ext = -2.0 * self.h_ext * self.time_func(t)*self.delta_t / self.H_BAR
                    elif "False" in self.custom_time_dep:
                        psi_ext=-2.0*self.h_ext*np.cos(self.freq*t)*self.delta_t/self.H_BAR
                    else:
                        print("Invalid selection for custom_time_dep parameter. Please enter True or False.")
                        with open(self.namevar,'a') as tempfile:
                            tempfile.write("Invalid selection for custom_time_dep parameter. Please enter True or False.\n")
                        break
                ext_instr_set=[]
                XX_instr_set=[]
                YY_instr_set=[]
                ZZ_instr_set=[]
                measure_set=[]
                for q in range(self.num_spins):
                    if self.ext_dir in "X":
                        ext_instr_set.append(Gate('RX', [q], angles=[psi_ext]))
                    elif self.ext_dir in "Y":
                        ext_instr_set.append(Gate('RY', [q], angles=[psi_ext]))
                    elif self.ext_dir in "Z":
                        ext_instr_set.append(Gate('RZ', [q], angles=[psi_ext]))
                psiX=-2.0*(self.Jx)*self.delta_t/self.H_BAR
                psiY=-2.0*(self.Jy)*self.delta_t/self.H_BAR
                psiZ=-2.0*(self.Jz)*self.delta_t/self.H_BAR

                for q in range(self.num_spins-1):
                    XX_instr_set.append(Gate('H',[q]))
                    XX_instr_set.append(Gate('H',[q+1]))
                    XX_instr_set.append(Gate('CNOT',[q, q+1]))
                    XX_instr_set.append(Gate('RZ', [q+1], angles=[psiX]))
                    XX_instr_set.append(Gate('CNOT',[q, q+1]))
                    XX_instr_set.append(Gate('H',[q]))
                    XX_instr_set.append(Gate('H',[q+1]))

                    YY_instr_set.append(Gate('RX',[q],angles=[-np.pi/2]))
                    YY_instr_set.append(Gate('RX',[q+1],angles=[-np.pi/2]))
                    YY_instr_set.append(Gate('CNOT',[q, q+1]))
                    YY_instr_set.append(Gate('RZ', [q+1], angles=[psiY]))
                    YY_instr_set.append(Gate('CNOT',[q, q+1]))
                    YY_instr_set.append(Gate('RX',[q],angles=[np.pi/2]))
                    YY_instr_set.append(Gate('RX',[q+1],angles=[np.pi/2]))

                    ZZ_instr_set.append(Gate('CNOT',[q, q+1]))
                    ZZ_instr_set.append(Gate('RZ', [q+1], angles=[psiZ]))
                    ZZ_instr_set.append(Gate('CNOT',[q, q+1]))

                if self.h_ext != 0:
                    P.add_instr(ext_instr_set)
                if self.Jx !=0:
                    P.add_instr(XX_instr_set)
                if self.Jy !=0:
                    P.add_instr(YY_instr_set)
                if self.Jz !=0:
                    P.add_instr(ZZ_instr_set)
            if "x" in self.observable_dir[i]:
                for q in range(self.num_qubits):
                    measure_set.append(Gate('H',[q]))
                P.add_instr(measure_set)
            elif "y" in self.observable_dir[i]:
                for q in range(self.num_qubits):
                    measure_set.append(Gate('RX',[q],angles=[-np.pi/2]))
                P.add_instr(measure_set)
        return P

    def generate_programs(self):
        programs = []
        #create programs for real_time evolution
        if (self.real_time == "True"):
            for j in range(0, self.steps+1):
                print("Generating timestep {} program".format(j))
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Generating timestep {} program\n".format(j))
                evolution_time = self.delta_t * j
                programs.append(self.heisenberg_evolution_program(evolution_time))
            self.programs_list=programs
        #create programs for imaginary time evolution
        else:
            from QITE import make_QITE_circ
            qcirc, energies = make_QITE_circ(self, qubits, beta, dbeta, domain, psi0, backend, regularizer)
            
        #convert to backend specific circuits if one is requested    
        if self.backend in "ibm":
            self.generate_ibm()
        if self.backend in "rigetti":
            self.generate_rigetti()
        if self.backend in "cirq":
            self.generate_cirq()


    def generate_ibm(self):
        tempcirclist=[]
        #convert from local circuits to IBM-specific circuit
        #IBM imports 
        import qiskit as qk
        from qiskit.tools.monitor import job_monitor
        from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
        from qiskit import Aer, IBMQ, execute
        from qiskit.providers.aer import noise
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.circuit import quantumcircuit
        from qiskit.circuit import Instruction
        qr=qk.QuantumRegister(self.num_spins, 'q')
        cr=qk.ClassicalRegister(self.num_spins, 'c')

        print("Creating IBM quantum circuit objects...")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("Creating IBM quantum circuit objects...\n")
        name=0
        for program in self.programs_list:
            propcirc = qk.QuantumCircuit(qr, cr)
            index=0
            for flip in self.initial_spins:
                if int(flip)==1:
                    propcirc.x(qr[index])
                    index+=1
                else: index+=1
            propcirc.barrier()
            for gate in program.gates:
                if "H" in gate.name:
                    propcirc.h(gate.qubits[0])
                elif "RZ" in gate.name:
                    propcirc.rz(gate.angles[0],gate.qubits[0])
                elif "RX" in gate.name:
                    propcirc.rx(gate.angles[0],gate.qubits[0])
                elif "CNOT" in gate.name:
                    propcirc.cx(gate.qubits[0],gate.qubits[1])
            propcirc.measure(qr,cr)
            tempcirclist.append(propcirc)
        self.ibm_circuits_list=tempcirclist
        print("IBM quantum circuit objects created")
        with open(self.namevar,'a') as tempfile:
            tempfile.write("IBM quantum circuit objects created\n")

        if "True" in self.compile:
            provider = qk.IBMQ.get_provider(group='open')
            device = provider.get_backend(self.device)
            #gather fidelity statistics on this device if you want to create a noise model for the simulator
            #properties = device.properties()
            #coupling_map = device.configuration().coupling_map
            #noise_model = NoiseModel.from_backend(device)
            #basis_gates = noise_model.basis_gates

            if self.compiler in "ds":
                temp=[]
                print("Compiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Compiling circuits...\n")
                for circuit in self.ibm_circuits_list:
                    compiled=ds_compile(circuit,self.backend)
                    temp.append(compiled)
                self.ibm_circuits_list=temp
                print("Circuits compiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits compiled successfully\n")
            elif self.compiler in "native":
                print("Transpiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Transpiling circuits...\n")
                temp=qk.compiler.transpile(self.ibm_circuits_list,backend=device,optimization_level=3)
                self.ibm_circuits_list=temp
                print("Circuits transpiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits transpiled successfully\n")
            elif self.compiler in "tket":
                from pytket.qiskit import qiskit_to_tk
                from pytket.backends.ibm import IBMQBackend, IBMQEmulatorBackend, AerBackend
                if self.device == "":
                    tket_backend = AerBackend()
                else: 
                    if self.QCQS in ["QC"]:
                        tket_backend = IBMQBackend(self.device)
                    else: 
                        tket_backend = IBMQEmulatorBackend(self.device)
                print("Compiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Compiling circuits...\n")
                for circuit in self.ibm_circuit_list:
                    tket_circ = qiskit_to_tk(c)
                    tket_backend.compile_circuit(tket_circ)
                    temp.append(tket_circ)
                self.ibm_circuits_list=temp
                print("Circuits compiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits compiled successfully\n")



    def generate_rigetti(self):
        import pyquil
        from pyquil.quil import Program
        from pyquil.gates import H, RX, RZ, CZ, RESET, MEASURE
        from pyquil.api import get_qc
        tempcirclist=[]
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
                if gate.name in "H":
                    p.inst(pyquil.gates.H(gate.qubits[0]))
                elif gate.name in "RZ":
                    p.inst(pyquil.gates.RZ(gate.angles[0],gate.qubits[0]))
                elif gate.name in "RX":
                    p.inst(pyquil.gates.RX(gate.angles[0],gate.qubits[0]))
                elif gate.name in "CNOT":
                    p.inst(pyquil.gates.CNOT(gate.qubits[0],gate.qubits[1]))
            for i in range(self.num_spins):
                p.inst(pyquil.gates.MEASURE(i,ro[i]))
            p.wrap_in_numshots_loop(self.shots)
            tempcirclist.append(p)
        self.rigetti_circuits_list=tempcirclist

        if "True" in self.compile:
            if self.QCQS in ["QS"]:
                qc=get_qc(self.device_choice, as_qvm=True)
            else:
                qc=get_qc(self.device_choice)
            qc.compiler.timeout = 20
            if self.default_compiler in "ds":
                temp=[]
                print("Compiling circuits...")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Compiling circuits...\n")
                for circuit in self.rigetti_circuits_list:
                    temp.append(ds_compile(circuit,self.backend,self.shots))
                self.rigetti_circuits_list=temp
                print("Circuits compiled successfully")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Circuits compiled successfully\n")
            elif self.default_compiler in "native":
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
                for circuit in self.ibm_circuit_list:
                    tket_circ = qiskit_to_tk(c)
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
        tempcirclist=[]
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
                elif gate.name in "RZ":
                    gate_list.append(cirq.rz(gate.angles[0])(qubit_list[gate.qubits[0]]))
                elif gate.name in "RX":
                    gate_list.append(cirq.rx(gate.angles[0])(qubit_list[gate.qubits[0]]))
                elif gate.name in "CNOT":
                    gate_list.append(cirq.CNOT(qubit_list[gate.qubits[0]],qubit_list[gate.qubits[1]]))
            for i in range(self.num_spins):
                gate_list.append(cirq.measure(qubit_list[i]))
            c.append(gate_list,strategy=cirq.InsertStrategy.EARLIEST)
            tempcirclist.append(c)
        self.cirq_circuits_list=tempcirclist
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


    def parameters(self):
        print("Current model parameters:\n\nH_BAR = {}\nJX = {}\nJY = {}\nJZ = {}\nh_ext = {}\next_dir = {}".format(self.H_BAR,self.JX,self.JY,self.JZ,self.h_ext,self.ext_dir))
        print("num_spins = {}\ninitial_spins = {}\ndelta_t = {}\nsteps = {}\nQCQS = {}\nshots = {}\nnoise_choice = {}".format(self.num_spins,self.initial_spins,self.delta_t,self.steps,self.QCQS,self.shots,self.noise_choice))
        print("device choice = {}\nplot_flag = {}\nfreq = {}\ntime_dep_flag = {}\ncustom_time_dep = {}\n".format(self.device_choice,self.plot_flag,self.freq,self.time_dep_flag,self.custom_time_dep)) 
        print("compile = {}\nauto_smart_compile = {}\ndefault_compiler = {}".format(self.compile,self.auto_smart_compile,self.default_compiler))
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
            ## Show available backends
            provider = qk.IBMQ.get_provider(group='open')
            provider.backends()

            #choose the device you would like to run on
            device = provider.get_backend(self.device)
            #gather fidelity statistics on this device if you want to create a noise model for the simulator
            properties = device.properties()
            coupling_map = device.configuration().coupling_map
            noise_model = NoiseModel.from_backend(device)
            basis_gates = noise_model.basis_gates

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
                        result_noise=execute(self.ibm_circuits_list,simulator,shots=self.shots).result()
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
                for j in range(self.num_spins):
                    avg_mag_sim = []
                    temp = []
                    i = 1
                    print("Post-processing qubit {} data".format(j+1))
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Post-processing qubit {} data\n".format(j+1))
                    for c in self.ibm_circuits_list:
                        result_dict = result_noise.get_counts(c)
                        temp.append(self.average_magnetization(result_dict, self.shots,j))
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

            elif self.QCQS in ["QC"]:
                #QUANTUM COMPUTER POST PROCESSING
                for j in range(self.num_spins):
                    results = job.result()        
                    avg_mag_qc = []
                    temp = []
                    i = 1
                    print("Post-processing qubit {} data".format(j+1))
                    with open(self.namevar,'a') as tempfile:
                        tempfile.write("Post-processing qubit {} data\n".format(j+1))
                    for c in self.ibm_circuits_list:
                            result_dict = results.get_counts(c)
                            temp.append(self.average_magnetization(result_dict, self.shots,j))
                            if i % (self.steps+1) == 0:
                                    avg_mag_qc.append(temp)
                                    temp = []
                            i += 1
                    
                    # QC
                    if "True" in self.plot_flag:
                        fig, ax = plt.subplots()
                        plt.plot(range(self.steps+1), avg_mag_qc[0])
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
                        plt.savefig("data/QC_result_qubit{}.png".format(j+1))
                        plt.close()
                    self.result_out_list.append(avg_mag_qc[0])
                    existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, self.num_spins))
                    np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,self.num_spins,len(existing)+1),avg_mag_qc[0])
                self.result_matrix=np.stack(self.result_out_list)           
                print("Done")
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Done\n")
        elif "rigetti" in self.backend:
            import pyquil
            from pyquil.quil import Program
            from pyquil.gates import H, RX, RZ, CZ, RESET, MEASURE
            from pyquil.api import get_qc
            print("Running Pyquil programs...")
            with open(self.namevar,'a') as tempfile:
                tempfile.write("Running Pyquil programs...\n")
            qc=get_qc(self.device_choice)
            results_list=[]
            first_ind=0
            #each circuit represents one timestep
            for circuit in self.rigetti_circuits_list:
                temp=qc.run(circuit)
                results_list.append(temp)

            for i in range(self.num_spins):
                print("Post-processing qubit {} data...".format(i+1))
                with open(self.namevar,'a') as tempfile:
                    tempfile.write("Post-Processing qubit {} data...\n".format(i+1))
                qubit_specific_row=np.zeros(len(results_list))
                for j in range(len(self.rigetti_circuits_list)):
                    results=results_list[j]

                    summation=0
                    for array in results:
                        summation+=(1-2*array[i])

                    summation=summation/len(results) #average over the number of shots

                    qubit_specific_row[j]=summation
                if first_ind==0:
                    self.result_matrix=qubit_specific_row
                    first_ind+=1
                else:
                    self.result_matrix=np.vstack((self.result_matrix,qubit_specific_row))
                if "y" in self.plot_flag:
                    plt.figure()
                    xaxis=np.linspace(0,self.steps,num=self.steps+1)
                    plt.plot(qubit_specific_row)
                    plt.xlabel("Simulation Timestep")
                    plt.ylabel("Average Magnetization")
                    plt.savefig("data/Result_qubit{}.png".format(i+1))
                    plt.close()
                existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(i+1,self.num_spins))
                np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(i+1,self.num_spins,len(existing)+1),qubit_specific_row)
            print("Done")
            with open(self.namevar,'a') as tempfile:
                tempfile.write("Done\n")
