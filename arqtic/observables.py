def system_magnetization(sim_object):
  """Compute average magnetization from results of qk.execution.
  Args:
  - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
  - shots (int): number of trials
  Return:
  - average_mag (float)
  """
    def system_mag_process(result:dict,shots:int)
		mag = 0
		for spin_str, count in result.items():
			spin_int = [1 - 2 * float(s) for s in spin_str]
			mag += (sum(spin_int) / len(spin_int)) * count
	  	average_mag = mag / shots
	  	return average_mag

	avg_mag = []
	temp = []
	i = 1
	print("Post-processing data")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Post-processing data\n")
	for c in sim_object.ibm_circuits_list:
		result_dict = sim_object.result_object.get_counts(c)
		temp.append(system_mag_process(result_dict, sim_object.shots))
		if i % (sim_object.steps+1) == 0:
			avg_mag.append(temp)
			temp = []
		i += 1

	if "True" in sim_object.plot_flag:
		fig, ax = plt.subplots()
		plt.plot(range(sim_object.steps+1), avg_mag[0])
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
		plt.savefig("data/Result.png")
		plt.close()
	sim_object.result_out_list.append(avg_mag[0])
	existing=glob.glob("data/System Average Magnetization Data, Qubits={}, num_*.txt".format(sim_object.num_spins))
	np.savetxt("data/System Average Magnetization Data, Qubits={}, num_{}.txt".format(sim_object.num_spins,len(existing)+1),avg_mag[0])
	sim_object.result_matrix=np.stack(sim_object.result_out_list)

	print("Done")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Done\n")
	return sim_object.result_matrix
		
		
            
def individual_magnetization(sim_object): 
		"""Compute average magnetization from results of qk.execution.
	  Args:
	  - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
	  - shots (int): number of trials
	  Return:
	  - average_mag (float)
	  """
	def individual_mag_process(result:dict,shots:int,qub:int)  
		mag = 0
		for spin_str, count in result.items():
			spin_int = [1 - 2 * float(spin_str[qub])]
			#print(spin_str)
			mag += (sum(spin_int) / len(spin_int)) * count
		average_mag = mag / shots
		return average_mag

	for j in range(sim_object.num_spins):
		avg_mag = []
		temp = []
		i = 1
		print("Post-processing qubit {} data".format(j+1))
		with open(sim_object.namevar,'a') as tempfile:
			tempfile.write("Post-processing qubit {} data\n".format(j+1))
		for c in sim_object.ibm_circuits_list:
			result_dict = sim_object.result_object.get_counts(c)
			temp.append(individual_mag_process(result_dict, sim_object.shots,j))
			if i % (sim_object.steps+1) == 0:
				avg_mag.append(temp)
				temp = []
			i += 1

		if "True" in sim_object.plot_flag:
			fig, ax = plt.subplots()
			plt.plot(range(sim_object.steps+1), avg_mag[0])
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
			plt.savefig("data/Result_qubit{}.png".format(j+1))
			plt.close()
		sim_object.result_out_list.append(avg_mag[0])
		existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, sim_object.num_spins))
		np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,sim_object.num_spins,len(existing)+1),avg_mag[0])
	sim_object.result_matrix=np.stack(sim_object.result_out_list)
	print("Done")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Done\n")

def energy(sim_object):
	#Alright general idea: gotta make that observables_axis happen. Actually maybe make a new thing called coefficients_list
	#or something to support having the hx, hy, or hz in there so it can differentiate when single qubit processing occurs
	num_pairs=sim_object.num_spins-1
	num_dirs=len(sim_object.observable_axis)

	def energy_process_twoqub(result:dict,shots:int,qub1:int,qub2:int)  
		energy1 = 0
		energy2 = 0
		for spin_str, count in result.items():
			spin_int1 = [1 - 2 * float(spin_str[qub1])]
			energy1 += (sum(spin_int1) / len(spin_int1)) * count
			spin_int2 = [1 - 2 * float(spin_str[qub2])]
			energy2 += (sum(spin_int2) / len(spin_int2)) * count
		average_energy1 = energy1 / shots
		average_energy2 = energy2 / shots
		return average_energy1*average_energy2

	def energy_process_onequb(result:dict,shots:int)  
		energy = 0
		for spin_str, count in result.items():
			spin_int = [1 - 2 * float(s) for s in spin_str]
			energy += (sum(spin_int) / len(spin_int)) * count
	  	average_energy = mag / shots
	  	return average_energy

	avg_en = []


	print("Post-processing data")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Post-processing data\n")

	i=1
	overall_temp = []
	summed_avg_en=np.zeros(sim_object.steps+1)
	coefficient_index=0
	#alright we have one full timeseries worth of circuits for every direction, right after each other
	#For each timestep, we need to sum coefficient*observable, so make a list of timestep arrays (one for each
	#direction, then element-wise sum them all
	for c in sim_object.ibm_circuits_list:
		temp=[]
		overall_temp=[]
		result_dict = sim_object.result_object.get_counts(c)
		if "Jx" in sim_object.coefficient_list[coefficient_index]:
			matching_coefficient=sim_object.Jx
		elif "Jy" in sim_object.coefficient_list[coefficient_index]:
			matching_coefficient=sim_object.Jy
		elif "Jz" == sim_object.coefficient_list[coefficient_index]:
			matching_coefficient=sim_object.Jz
		elif sim_object.coefficient_list[coefficient_index] in ["hx","hy","hz"]:
			#need to figure out how to support time dependent h(t) in multiplication here
			matching_coefficient=0 #placeholder
		
		if sim_object.coefficient_list[coefficient_index] in ["Jx","Jy","Jz"]:
			for z in range(num_pairs):
				temp.append(energy_process_twoqub(result_dict, sim_object.shots,z,z+1))
				overall_temp.append(np.sum(temp))
		elif sim_object.coefficient_list[coefficient_index] in ["hx","hy","hz"]:
			overall_temp.append(energy_process_onequb(result_dict,sim_object.shots))

		if i % (sim_object.steps+1) == 0:
			summed_avg_en+=matching_coefficient*np.array(overall_temp) 
			#every time it counts the length of the timeseries, that means a whole direction has finished.
			#then sum each direction with each other elementwise to get sum of each direction at every timestep
			overall_temp = []
			coefficient_index+=1
		i += 1

	if "True" in sim_object.plot_flag:
		fig, ax = plt.subplots()
		plt.plot(range(sim_object.steps+1), summed_avg_en)
		plt.xlabel("Simulation Timestep",fontsize=14)
		plt.ylabel("Average Energy",fontsize=14)
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
		plt.savefig("data/Result_qubit{}.png".format(j+1))
		plt.close()
	sim_object.result_out_list.append(summed_avg_en)
	existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, sim_object.num_spins))
	np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,sim_object.num_spins,len(existing)+1),summed_avg_en)
	sim_object.result_matrix=np.stack(sim_object.result_out_list)
	print("Done")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Done\n")

def staggered_mag(sim_object):
	def staggered_magnetization_process(result: dict, shots: int):

        sm_val = 0

        for spin_str, count in result.items():

            spin_int = [1 - 2 * float(s) for s in spin_str]

            for i in range(len(spin_int)):

                spin_int[i] = spin_int[i]*(-1)**i

            sm_val += (sum(spin_int) / len(spin_int)) * count

        average_sm = sm_val/shots

        return average_sm

	stag_mag = []
	temp = []
	i = 1
	print("Post-processing data")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Post-processing data\n")
	for c in sim_object.ibm_circuits_list:
		result_dict = sim_object.result_object.get_counts(c)
		temp.append(staggered_magnetization_process(result_dict, sim_object.shots))
		if i % (sim_object.steps+1) == 0:
			stag_mag.append(temp)
			temp = []
		i += 1

	if "True" in sim_object.plot_flag:
		fig, ax = plt.subplots()
		plt.plot(range(sim_object.steps+1), stag_mag[0])
		plt.xlabel("Simulation Timestep",fontsize=14)
		plt.ylabel("Staggered Magnetization",fontsize=14)
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
		plt.savefig("data/Result.png")
		plt.close()
	sim_object.result_out_list.append(stag_mag[0])
	existing=glob.glob("data/Staggered Magnetization Data, Qubits={}, num_*.txt".format(sim_object.num_spins))
	np.savetxt("data/Staggered Magnetization Data, Qubits={}, num_{}.txt".format(sim_object.num_spins,len(existing)+1),stag_mag[0])
	sim_object.result_matrix=np.stack(sim_object.result_out_list)

	print("Done")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Done\n")
	return sim_object.result_matrix
