def local_magnetization(N, result: dict, shots: int, qub: int):
    """Compute average magnetization from results of qk.execution.
    Args:
    - N: number of spins
    - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
    - shots (int): number of trials
    Return:
    - average_mag (float)
    """
    mag = 0
    q_idx = N - qub -1
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(spin_str[q_idx])]
        mag += (sum(spin_int) / len(spin_int)) * count
    average_mag = mag / shots
    return average_mag

def staggered_magnetization(result: dict, shots: int):
    sm_val = 0
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(s) for s in spin_str]
        for i in range(len(spin_int)):
            spin_int[i] = spin_int[i]*(-1)**i
        sm_val += (sum(spin_int) / len(spin_int)) * count
    average_sm = sm_val/shots
    return average_sm

def system_magnetization(result: dict, shots: int):
    mag_val = 0
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(s) for s in spin_str]
        mag_val += (sum(spin_int) / len(spin_int)) * count
    average_mag = mag_val/shots
    return average_mag

def excitation_displacement(N, result: dict, shots: int):
    dis = 0
    for qub in range(N):
        z = local_magnetization(N,result, shots, qub)
        dis += qub*((1.0 - z)/2.0)
    return dis



#Rigetti
#results list has length L+1 with L=number of timesteps
#for each element in the list (corresponding to a single timestep), it is a list of shot results (S sublists each consisting of N numbers), 
#S = number of shots, N = number of spins 
#Again, these function calls get called on a circuit-by-circuit basis, so once for each timestep

def local_magnetization_rigetti(N, result, shots: int, qub: int):
    """Compute average magnetization from results of qk.execution.
    Args:
    - N: number of spins
    - result (list): a nested list with the results of each shot
    - shots (int): number of trials
    Return:
    - average_mag (float)
    """
    mag = 0
    q_idx = qub
    for elem in result:
        spin_int = [1 - 2 * float(elem[q_idx])]
        mag += (sum(spin_int) / len(spin_int))
    average_mag = mag / shots
    return average_mag

def staggered_magnetization_rigetti(result, shots: int):
    sm_val = 0
    for elem in result:
        spin_int = [1 - 2 * float(s) for s in elem]
        for i in range(len(spin_int)):
            spin_int[i] = spin_int[i]*(-1)**i
        sm_val += (sum(spin_int) / len(spin_int))
    average_sm = sm_val/shots
    return average_sm

def system_magnetization_rigetti(result, shots: int):
    mag_val = 0
    for elem in result:
        spin_int = [1 - 2 * float(s) for s in elem]
        mag_val += (sum(spin_int) / len(spin_int))
    average_mag = mag_val/shots
    return average_mag

def excitation_displacement_rigetti(N, result, shots: int):
    dis = 0
    for qub in range(N):
        z = local_magnetization_rigetti(N,result, shots, qub)
        dis += qub*((1.0 - z)/2.0)
    return dis