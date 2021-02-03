def system_magnetization(sim_object):
  """Compute average magnetization from results of qk.execution.
  Args:
  - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
  - shots (int): number of trials
  Return:
  - average_mag (float)
  """
    def system_mag_process(result,shots)
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
			avg_mag_sim.append(temp)
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
	def individual_mag_process(result,shots,qub)  
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
				avg_mag_sim.append(temp)
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
	num_pairs=sim_object.num_spins-1
	num_dirs=len(sim_object.direction_list)

	def energy_process(result,shots,qub1,qub2)  
		energy = 0
		for spin_str, count in result.items():
			multiplied = float(spin_str[qub1]) * float(spin_str[qub2])
			energy += energy * count
		average_energy = energy / shots
		return average_energy

	avg_en = []


	print("Post-processing data")
	with open(sim_object.namevar,'a') as tempfile:
		tempfile.write("Post-processing data\n")

	i=1
	overall_temp = []
	summed_avg_en=np.zeros(sim_object.steps+1)
	for c in sim_object.ibm_circuits_list:
		temp=[]
		result_dict = sim_object.result_object.get_counts(c)
		for z in range(num_pairs):
			temp.append(energy_process(result_dict, sim_object.shots,z,z+1))

		overall_temp.append(np.sum(temp))
		if i % (sim_object.steps+1) == 0:
			summed_avg_en+=np.array(overall_temp) 
			#every time it counts the length of the timeseries, that means a whole direction has finished.
			#then sum each direction with each other elementwise to get sum of each direction at every timestep
			overall_temp = []
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
