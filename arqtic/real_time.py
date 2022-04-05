import numpy as np
from arqtic.program import Program, Gate

def heisenberg_evolution_program(sim_obj, evol_time): #creates evolution program
    N = sim_obj.num_spins #this is the case for 1D systems
    dt = sim_obj.delta_t
    H_BAR = sim_obj.H_BAR
    prop_steps = int(evol_time/dt)
    P = Program(N)
    
    #prepare coupling term arrays
    if (len(sim_obj.Jx) > 0):
        if (len(sim_obj.Jx) == 1):
            Jx = np.full(sim_obj.num_spins-1, float(sim_obj.Jx[0]))
        elif (sim_obj.Jx[0] == "random"):
            lower = float(sim_obj.Jx[1])
            upper = float(sim_obj.Jx[2])
            Jx = np.random.uniform(lower,upper,sim_obj.num_spins-1)
        else:
            Jx = np.asarray([float(x) for x in sim_obj.Jx])
    else: Jx = []
    if (len(sim_obj.Jy) > 0):
        if (len(sim_obj.Jy) == 1):
            Jy = np.full(sim_obj.num_spins-1, float(sim_obj.Jy[0]))
        elif (sim_obj.Jy[0] == "random"):
            lower = float(sim_obj.Jy[1])
            upper = float(sim_obj.Jy[2])
            Jy = np.random.uniform(lower,upper,sim_obj.num_spins-1)
        else:
            Jy = np.asarray([float(x) for x in sim_obj.Jy])
    else: Jy = []
    if (len(sim_obj.Jz) > 0):
        if (len(sim_obj.Jz) == 1):
            Jz = np.full(sim_obj.num_spins-1, float(sim_obj.Jz[0]))
        elif (sim_obj.Jz[0] == "random"):
            lower = float(sim_obj.Jz[1])
            upper = float(sim_obj.Jz[2])
            Jz = np.random.uniform(lower,upper,sim_obj.num_spins-1)
        else:
            Jz = np.asarray([float(x) for x in sim_obj.Jz])
    else: Jz = []

    #prepare external magnetic field term arrays 
    if (len(sim_obj.hx) > 0):
        if (len(sim_obj.hx) == 1):
            hx = np.full(sim_obj.num_spins, float(sim_obj.hx[0]))
        elif (sim_obj.hx[0] == "random"):
            lower = float(sim_obj.hx[1])
            upper = float(sim_obj.hx[2])
            hx = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hx = np.asarray([float(x) for x in sim_obj.hx])
    else: hx = []
    if (len(sim_obj.hy) > 0):
        if (len(sim_obj.hy) == 1):
            hy = np.full(sim_obj.num_spins, float(sim_obj.hy[0]))
        elif (sim_obj.hy[0] == "random"):
            lower = float(sim_obj.hy[1])
            upper = float(sim_obj.hy[2])
            hy = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hy = np.asarray([float(x) for x in sim_obj.hy])
    else: hy = []
    if (len(sim_obj.hz) > 0):
        if (len(sim_obj.hz) == 1):
            hz = np.full(sim_obj.num_spins, float(sim_obj.hz[0]))
        elif (sim_obj.hz[0] == "random"):
            lower = float(sim_obj.hz[1])
            upper = float(sim_obj.hz[2])
            hz = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hz = np.asarray([float(x) for x in sim_obj.hz])
    else: hz = []

    #time dependence
    if (sim_obj.time_dep_flag == "True"):
        if(len(sim_obj.td_Jx_func) > 0):
            func = []
            func.append(sim_obj.td_Jx_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_Jx_func) - 1):
                func.append(float(sim_obj.td_Jx_func[1+p]))
            td_Jx_func = func
        if(len(sim_obj.td_Jy_func) > 0):
            func = []
            func.append(sim_obj.td_Jy_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_Jy_func) - 1):
                func.append(float(sim_obj.td_Jy_func[1+p]))
            td_Jy_func = func
        if(len(sim_obj.td_Jz_func) > 0):
            func = []
            func.append(sim_obj.td_Jz_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_Jz_func) - 1):
                func.append(float(sim_obj.td_Jz_func[1+p]))
            td_Jz_func = func
        if(len(sim_obj.td_hx_func) > 0):
            func = []
            func.append(sim_obj.td_hx_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_hx_func) - 1):
                func.append(float(sim_obj.td_hx_func[1+p]))
            td_hx_func = func
        if(len(sim_obj.td_hy_func) > 0):
            func = []
            func.append(sim_obj.td_hy_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_hy_func) - 1):
                func.append(float(sim_obj.td_hy_func[1+p]))
            td_hy_func = func
        if(len(sim_obj.td_hz_func) > 0):
            func = []
            func.append(sim_obj.td_hz_func[0]) #time-dependent function name
            for p in range(len(sim_obj.td_hz_func) - 1):
                func.append(float(sim_obj.td_hz_func[1+p]))
            td_hz_func = func
    
    
    #time-independent Hamiltonian
    if (len(Jx) > 0):
        theta_Jx = 2.0*Jx*dt/H_BAR
    if (len(Jy) > 0):
        theta_Jy = 2.0*Jy*dt/H_BAR
    if (len(Jz) > 0):
        theta_Jz = 2.0*Jz*dt/H_BAR
    if (len(hx) > 0):
        theta_hx = 2.0*hx*dt/H_BAR
    if (len(hy) > 0):
        theta_hy = 2.0*hy*dt/H_BAR
    if (len(hz) > 0):
        theta_hz = 2.0*hz*dt/H_BAR

    for step in range(prop_steps):
        #for time-dependent Hamiltonians
        if (sim_obj.time_dep_flag == "True"):
            t = (step + 0.5) * dt
            if (sim_obj.custom_time_dep == "True"):
                ###needs to be implemented###
                raise Error(f'Custom time-dependence not yet implemented. Use sin or cos.')
            else:
                if (len(td_Jx_func) > 0):
                    func_name = td_Jx_func[0]
                    freq = td_Jx_func[1]
                    if (func_name == "sin"):
                        theta_Jx = 2.0*Jx*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jx = 2.0*Jx*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(td_Jy_func) > 0):
                    func_name = td_Jy_func[0]
                    freq = td_Jy_func[1]
                    if (func_name == "sin"):
                        theta_Jy = 2.0*Jy*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jy = 2.0*Jy*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(td_Jz_func) > 0):
                    func_name = td_Jz_func[0]
                    freq = td_Jz_func[1]
                    if (func_name == "sin"):
                        theta_Jz = 2.0*Jz*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jz = 2.0*Jz*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(td_hx_func) > 0):
                    func_name = td_hx_func[0]
                    freq = td_hx_func[1]
                    if (func_name == "sin"):
                        theta_hx = 2.0*hx*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hx = 2.0*hx*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(td_hy_func) > 0):
                    func_name = td_hy_func[0]
                    freq = td_hy_func[1]
                    if (func_name == "sin"):
                        theta_hy = 2.0*hy*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hy = 2.0*hy*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(td_hz_func) > 0):
                    func_name = td_hz_func[0]
                    freq = td_hz_func[1]
                    if (func_name == "sin"):
                        theta_hz = 2.0*hz*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hz = 2.0*hz*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')

        #add coupling term instruction sets
        if (len(Jx) >0):
            for q in range(N-1):
                Jx_instr_set=[]
                Jx_instr_set.append(Gate([q], 'H'))
                Jx_instr_set.append(Gate([q+1], 'H'))
                Jx_instr_set.append(Gate([q, q+1], 'CNOT'))
                Jx_instr_set.append(Gate([q+1], 'RZ', angles=[theta_Jx[q]]))
                Jx_instr_set.append(Gate([q, q+1], 'CNOT'))
                Jx_instr_set.append(Gate([q], 'H',))
                Jx_instr_set.append(Gate([q+1], 'H',))
                P.add_instr(Jx_instr_set)
        if (len(Jy) >0):
            for q in range(N-1):
                Jy_instr_set=[]
                Jy_instr_set.append(Gate([q],'RX',angles=[-np.pi/2]))
                Jy_instr_set.append(Gate([q+1],'RX',angles=[-np.pi/2]))
                Jy_instr_set.append(Gate([q, q+1],'CNOT'))
                Jy_instr_set.append(Gate([q+1], 'RZ', angles=[theta_Jy[q]]))
                Jy_instr_set.append(Gate([q, q+1], 'CNOT'))
                Jy_instr_set.append(Gate([q],'RX',angles=[np.pi/2]))
                Jy_instr_set.append(Gate([q+1],'RX',angles=[np.pi/2]))
                P.add_instr(Jy_instr_set)
        if (len(Jz) >0):
            for q in range(N-1):
                Jz_instr_set=[]
                Jz_instr_set.append(Gate([q, q+1], 'CNOT'))
                Jz_instr_set.append(Gate([q+1], 'RZ', angles=[theta_Jz[q]]))
                Jz_instr_set.append(Gate([q, q+1], 'CNOT'))
                P.add_instr(Jz_instr_set)

        #add external magnetic field instruction sets
        if (len(hx) > 0):
            for q in range(N):
                hx_instr_set = [Gate([q], 'RX', angles=[theta_hx[q]])]
                P.add_instr(hx_instr_set)
        if (len(hy) > 0):
            for q in range(N):
                hy_instr_set = [Gate([q], 'RY', angles=[theta_hy[q]])]
                P.add_instr(hy_instr_set)
        if (len(hz) > 0):
            for q in range(N):
                hz_instr_set = [Gate([q], 'RZ', angles=[theta_hz[q]])]
                P.add_instr(hz_instr_set)

    return P



def heisenberg2D_evolution_program(sim_obj, evol_time): #creates evolution program
    N = sim_obj.num_spins #total number of spins in 2D systems
    Nrows = sim_obj.Nrows #number of rows in 2D lattice
    Ncols = sim_obj.Ncols #number of columns in 2D lattice
    dt = sim_obj.delta_t
    H_BAR = sim_obj.H_BAR
    prop_steps = int(evol_time/dt)
    P = Program(N)
    
    #assume coupling strengths along each axis will be uniform across all qubits
    #though they can vary with time
    #assign initial values for coupling strengths
    if (len(sim_obj.Jx) > 0):
        Jx_list = np.asarray([float(x) for x in sim_obj.Jx])
        #Jx = Jx_list[0] #assume uniform coupling for all pairs
    else: Jx = []
    if (len(sim_obj.Jy) > 0):
        Jy_list = np.asarray([float(x) for x in sim_obj.Jy])
        #Jy = Jy_list[0] #assume uniform coupling for all pairs
    else: Jy = []
    if (len(sim_obj.Jz) > 0):
        Jz_list = np.asarray([float(x) for x in sim_obj.Jz])
        #Jz = Jz_list[0] #assume uniform coupling for all pairs
    else: Jz = []
        
    #external field terms can vary across the different qubits
    #assign arrays for the initial values of the external fields 
    if (len(sim_obj.hx) > 0):
        if (len(sim_obj.hx) == 1):
            hx = np.full(sim_obj.num_spins, float(sim_obj.hx[0]))
        elif (sim_obj.hx[0] == "random"):
            lower = float(sim_obj.hx[1])
            upper = float(sim_obj.hx[2])
            hx = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hx = np.asarray([float(x) for x in sim_obj.hx])
    else: hx = []
    if (len(sim_obj.hy) > 0):
        if (len(sim_obj.hy) == 1):
            hy = np.full(sim_obj.num_spins, float(sim_obj.hy[0]))
        elif (sim_obj.hy[0] == "random"):
            lower = float(sim_obj.hy[1])
            upper = float(sim_obj.hy[2])
            hy = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hy = np.asarray([float(x) for x in sim_obj.hy])
    else: hy = []
    if (len(sim_obj.hz) > 0):
        if (len(sim_obj.hz) == 1):
            hz = np.full(sim_obj.num_spins, float(sim_obj.hz[0]))
        elif (sim_obj.hz[0] == "random"):
            lower = float(sim_obj.hz[1])
            upper = float(sim_obj.hz[2])
            hz = np.random.uniform(lower,upper,sim_obj.num_spins)
        else:
            hz = np.asarray([float(x) for x in sim_obj.hz])
    else: hz = []
        
    #for time-dependent terms
    if (sim_obj.time_dep_flag == "True"):
        if(len(sim_obj.td_Jx_func) > 0):
            func_name = sim_obj.td_Jx_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_Jx_func = ["sin", sim_obj.td_Jx_func[1]]
            elif (func_name == "cos"):
                td_Jx_func = ["cos", sim_obj.td_Jx_func[1]]
            elif (func_name == "linear"):
                increment = (sim_obj.td_Jx_func[1] - Jx[0])/prop_steps
                td_Jx_func = ["linear", increment]
            else 
                raise Error(f'Unknown time-dependent function for Jx: {func_name}')
        if(len(sim_obj.td_Jy_func) > 0):
            func_name = sim_obj.td_Jy_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_Jy_func = ["sin", sim_obj.td_Jy_func[1]]
            elif (func_name == "cos"):
                td_Jy_func = ["cos", sim_obj.td_Jy_func[1]]
            elif (func_name == "linear"):
                increment = (sim_obj.td_Jy_func[1] - Jy[0])/prop_steps
                td_Jy_func = ["linear", increment]
            else 
                raise Error(f'Unknown time-dependent function for Jy: {func_name}')
        if(len(sim_obj.td_Jz_func) > 0):
            func_name = sim_obj.td_Jz_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_Jz_func = ["sin", sim_obj.td_Jz_func[1]]
            elif (func_name == "cos"):
                td_Jz_func = ["cos", sim_obj.td_Jz_func[1]]
            elif (func_name == "linear"):
                increment = (sim_obj.td_Jz_func[1] - Jz[0])/prop_steps
                td_Jz_func = ["linear", increment]
            else 
                raise Error(f'Unknown time-dependent function for Jz: {func_name}')
        if(len(sim_obj.td_hx_func) > 0):
            func_name = sim_obj.td_hx_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_hx_func = ["sin", sim_obj.td_hx_func[1]]
            elif (func_name == "cos"):
                td_hx_func = ["cos", sim_obj.td_hx_func[1]]
            elif (func_name == "linear"):
                final_vals = np.asarray(sim_obj.td_hx_func[1])
                if (len(final_vals) == 1):
                    final_vals = np.full(sim_obj.num_spins, float(final_vals[0]))
                increments = (final_vals - hx)/prop_steps
                td_hx_func = ["linear", increments]
            else 
                raise Error(f'Unknown time-dependent function for hx: {func_name}')
        if(len(sim_obj.td_hy_func) > 0):
            func_name = sim_obj.td_hy_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_hy_func = ["sin", sim_obj.td_hy_func[1]]
            elif (func_name == "cos"):
                td_hy_func = ["cos", sim_obj.td_hy_func[1]]
            elif (func_name == "linear"):
                final_vals = np.asarray(sim_obj.td_hy_func[1])
                if (len(final_vals) == 1):
                    final_vals = np.full(sim_obj.num_spins, float(final_vals[0]))
                increments = (final_vals - hy)/prop_steps
                td_hy_func = ["linear", increments]
            else 
                raise Error(f'Unknown time-dependent function for hy: {func_name}')
        if(len(sim_obj.td_hz_func) > 0):
            func_name = sim_obj.td_hz_func[0] #time-dependent function name
            if (func_name == "sin"):
                td_hz_func = ["sin", sim_obj.td_hz_func[1]]
            elif (func_name == "cos"):
                td_hz_func = ["cos", sim_obj.td_hz_func[1]]
            elif (func_name == "linear"):
                final_vals = np.asarray(sim_obj.td_hz_func[1])
                if (len(final_vals) == 1):
                    final_vals = np.full(sim_obj.num_spins, float(final_vals[0]))
                increments = (final_vals - hz)/prop_steps
                td_hz_func = ["linear", increments]
            else 
                raise Error(f'Unknown time-dependent function for hz: {func_name}')


    #time-independent Hamiltonian
    if (sim_obj.time_dep_flag == "False"):
        if (len(Jx) > 0):
            theta_Jx = 2.0*Jx*dt/H_BAR
        if (len(Jy) > 0):
            theta_Jy = 2.0*Jy*dt/H_BAR
        if (len(Jz) > 0):
            theta_Jz = 2.0*Jz*dt/H_BAR
        if (len(hx) > 0):
            theta_hx = 2.0*hx*dt/H_BAR
        if (len(hy) > 0):
            theta_hy = 2.0*hy*dt/H_BAR
        if (len(hz) > 0):
            theta_hz = 2.0*hz*dt/H_BAR       

    #loop over time-steps
    for step in range(prop_steps): 
        #compute time-dependent parameters
        if (sim_obj.time_dep_flag == "True"):
            t = step*dt
            if (len(td_Jx_func) > 0):
                    func_name = td_Jx_func[0]
                    if (func_name == "sin"):
                        freq = td_Jx_func[1]
                        theta_Jx = 2.0*Jx[0]*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_Jx_func[1]
                        theta_Jx = 2.0*Jx[0]*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increment = td_Jx_func[1]
                        val = Jx[0] + increment*step
                        theta_Jx = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
            if (len(td_Jy_func) > 0):
                    func_name = td_Jy_func[0]
                    if (func_name == "sin"):
                        freq = td_Jy_func[1]
                        theta_Jy = 2.0*Jy[0]*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_Jy_func[1]
                        theta_Jy = 2.0*Jy[0]*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increment = td_Jy_func[1]
                        val = Jy[0] + increment*step
                        theta_Jy = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jy: {func_name}')
            if (len(td_Jz_func) > 0):
                    func_name = td_Jz_func[0]
                    if (func_name == "sin"):
                        freq = td_Jz_func[1]
                        theta_Jz = 2.0*Jz[0]*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_Jz_func[1]
                        theta_Jz = 2.0*Jz[0]*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increment = td_Jz_func[1]
                        val = Jz[0] + increment*step
                        theta_Jz = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jz: {func_name}')
                        
            if (len(td_hx_func) > 0):
                    func_name = td_hx_func[0]
                    if (func_name == "sin"):
                        freq = td_hx_func[1]
                        theta_hx = 2.0*hx*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_hx_func[1]
                        theta_hx = 2.0*hx*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increments = td_hx_func[1]
                        val = hx + increments*step
                        theta_hx = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for hx: {func_name}')
            if (len(td_hy_func) > 0):
                    func_name = td_hy_func[0]
                    if (func_name == "sin"):
                        freq = td_hy_func[1]
                        theta_hy = 2.0*hy*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_hy_func[1]
                        theta_hy = 2.0*hy*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increments = td_hy_func[1]
                        val = hy + increments*step
                        theta_hy = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for hy: {func_name}')
            if (len(td_hz_func) > 0):
                    func_name = td_hz_func[0]
                    if (func_name == "sin"):
                        freq = td_hz_func[1]
                        theta_hz = 2.0*hz*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        freq = td_hz_func[1]
                        theta_hz = 2.0*hz*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "linear"):
                        increments = td_hz_func[1]
                        val = hz + increments*step
                        theta_hz = 2.0*val*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for hz: {func_name}')
               
        #initialize coupling arrays for each time-step
        Jx_coupling_array = []
        Jy_coupling_array = []
        Jz_coupling_array = []
        hx_coupling_array = []
        hy_coupling_array = []
        hz_coupling_array = []
        #prepare coupling term arrays, which are an array of arrays of the form [J,q1,q2] or [h,q]
        #loop over qubits in 2D lattice
        for r in range(Nrows):
            for c in range(Ncols):
                ref_idx = r*Nrows + c #index of reference qubit
                #add coupling terms for all 4 directions around reference qubit
                #couple to qubit above
                if (r>0):
                    if (len(Jx) > 0):
                        Jx_coupling_array.append([theta_Jx, ref_idx, ref_idx-Ncols])
                    if (len(Jy) > 0):
                        Jy_coupling_array.append([theta_Jy, ref_idx, ref_idx-Ncols])
                    if (len(Jz) > 0):
                        Jz_coupling_array.append([theta_Jz, ref_idx, ref_idx-Ncols])
                if (r == 0 and pbc == "True"):
                    if (len(Jx) > 0):
                        Jx_coupling_array.append([theta_Jx, ref_idx, Ncols*(Nrows-1)+c])
                    if (len(Jy) > 0):
                        Jy_coupling_array.append([theta_Jy, ref_idx, Ncols*(Nrows-1)+c])
                    if (len(Jz) > 0):
                        Jz_coupling_array.append([theta_Jz, ref_idx, Ncols*(Nrows-1)+c])  
                #couple to qubit below
                if (r<(Nrows-1)):
                    if (len(Jx) > 0):
                        Jx_coupling_array.append([theta_Jx, ref_idx, ref_idx+Ncols])
                    if (len(Jy) > 0):
                        Jy_coupling_array.append([theta_Jy, ref_idx, ref_idx+Ncols])
                    if (len(Jz) > 0):
                        Jz_coupling_array.append([theta_Jz, ref_idx, ref_idx+Ncols])
                if (r == (Nrows-1) and pbc == "True"):
                    if (len(Jx) > 0):
                        Jx_coupling_array.append([theta_Jx, ref_idx, c])
                    if (len(Jy) > 0):
                        Jy_coupling_array.append([theta_Jy, ref_idx, c])
                    if (len(Jz) > 0):
                        Jz_coupling_array.append([theta_Jz, ref_idx, c])  
                #couple to qubit to right
                #couple to qubit to left
            
    

    
    
