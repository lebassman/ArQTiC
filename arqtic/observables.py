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

