import numpy as np
from arqtic.program import Program, Gate

def heisenberg_evolution_program(sim_obj, evol_time): #creates evolution program
    N = sim_obj.num_spins
    dt = sim_obj.delta_t
    H_BAR = sim_obj.H_BAR
    prop_steps = int(evol_time/dt)
    P = Program(N)
    if (len(sim_obj.Jx) > 0):
        theta_Jx = 2.0*sim_obj.Jx*dt/H_BAR
    if (len(sim_obj.Jy) > 0):
        theta_Jy = 2.0*sim_obj.Jy*dt/H_BAR
    if (len(sim_obj.Jz) > 0):
        theta_Jz = 2.0*sim_obj.Jz*dt/H_BAR
    if (len(sim_obj.hx) > 0):
        theta_hx = 2.0*sim_obj.hx*dt/H_BAR
    if (len(sim_obj.hy) > 0):
        theta_hy = 2.0*sim_obj.hy*dt/H_BAR
    if (len(sim_obj.hz) > 0):
        theta_hz = 2.0*sim_obj.hz*dt/H_BAR

    for step in range(prop_steps):
        #for time-dependent Hamiltonians
        if (sim_obj.time_dep_flag == "True"):
            t = (step + 0.5) * dt
            if (sim_obj.custom_time_dep == "True"):
                ###needs to be implemented###
                raise Error(f'Custom time-dependence not yet implemented. Use sin or cos.')
            else:
                if (len(sim_obj.td_Jx_func) > 0):
                    func_name = sim_obj.td_Jx_func[0]
                    freq = sim_obj.td_Jx_func[1]
                    if (func_name == "sin"):
                        theta_Jx = 2.0*sim_obj.Jx*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jx = 2.0*sim_obj.Jx*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(sim_obj.td_Jy_func) > 0):
                    func_name = sim_obj.td_Jy_func[0]
                    freq = sim_obj.td_Jy_func[1]
                    if (func_name == "sin"):
                        theta_Jy = 2.0*sim_obj.Jy*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jy = 2.0*sim_obj.Jy*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(sim_obj.td_Jz_func) > 0):
                    func_name = sim_obj.td_Jz_func[0]
                    freq = sim_obj.td_Jz_func[1]
                    if (func_name == "sin"):
                        theta_Jz = 2.0*sim_obj.Jz*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_Jz = 2.0*sim_obj.Jz*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(sim_obj.td_hx_func) > 0):
                    func_name = sim_obj.td_hx_func[0]
                    freq = sim_obj.td_hx_func[1]
                    if (func_name == "sin"):
                        theta_hx = 2.0*sim_obj.hx*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hx = 2.0*sim_obj.hx*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(sim_obj.td_hy_func) > 0):
                    func_name = sim_obj.td_hy_func[0]
                    freq = sim_obj.td_hy_func[1]
                    if (func_name == "sin"):
                        theta_hy = 2.0*sim_obj.hy*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hy = 2.0*sim_obj.hy*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')
                if (len(sim_obj.td_hz_func) > 0):
                    func_name = sim_obj.td_hz_func[0]
                    freq = sim_obj.td_hz_func[1]
                    if (func_name == "sin"):
                        theta_hz = 2.0*sim_obj.hz*np.sin(2*np.pi*freq*t)*dt/H_BAR
                    elif (func_name == "cos"):
                        theta_hz = 2.0*sim_obj.hz*np.cos(2*np.pi*freq*t)*dt/H_BAR
                    else:
                        raise Error(f'Unknown time-dependent function for Jx: {func_name}')

        #add coupling term instruction sets
        if (len(sim_obj.Jx) >0):
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
        if (len(sim_obj.Jy) >0):
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
        if (len(sim_obj.Jz) >0):
            for q in range(N-1):
                Jz_instr_set=[]
                Jz_instr_set.append(Gate([q, q+1], 'CNOT'))
                Jz_instr_set.append(Gate([q+1], 'RZ', angles=[theta_Jz[q]]))
                Jz_instr_set.append(Gate([q, q+1], 'CNOT'))
                P.add_instr(Jz_instr_set)

        #add external magnetic field instruction sets
        if (len(sim_obj.hx) > 0):
            for q in range(N):
                hx_instr_set = [Gate([q], 'RX', angles=[theta_hx[q]])]
                P.add_instr(hx_instr_set)
        if (len(sim_obj.hy) > 0):
            for q in range(N):
                hy_instr_set = [Gate([q], 'RY', angles=[theta_hy[q]])]
                P.add_instr(hy_instr_set)
        if (len(sim_obj.hz) > 0):
            for q in range(N):
                hz_instr_set = [Gate([q], 'RZ', angles=[theta_hz[q]])]
                P.add_instr(hz_instr_set)

    return P

