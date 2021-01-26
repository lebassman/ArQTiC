def system_magnetization(result,shots):
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

	if "QS" in sim_object.QCQS:
		avg_mag_sim = []
		temp = []
		i = 1
		print("Post-processing data")
		with open(sim_object.namevar,'a') as tempfile:
			tempfile.write("Post-processing qubit\n")
		for c in sim_object.ibm_circuits_list:
			result_dict = result_noise.get_counts(c)
			temp.append(system_mag_process(result, sim_object.shots))
			if i % (sim_object.steps+1) == 0:
				avg_mag_sim.append(temp)
				temp = []
			i += 1

		if "True" in sim_object.plot_flag:
			fig, ax = plt.subplots()
			plt.plot(range(sim_object.steps+1), avg_mag_sim[0])
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
			plt.savefig("data/Simulator_result.png")
			plt.close()
		sim_object.result_out_list.append(avg_mag_sim[0])
		existing=glob.glob("data/System Average Magnetization Data, Qubits={}, num_*.txt".format(sim_object.num_spins))
		np.savetxt("data/System Average Magnetization Data, Qubits={}, num_{}.txt".format(sim_object.num_spins,len(existing)+1),avg_mag_sim[0])
		sim_object.result_matrix=np.stack(sim_object.result_out_list)
		print("Done")
		with open(sim_object.namevar,'a') as tempfile:
			tempfile.write("Done\n")
	if "QC" in sim_object.QCQS:
        for j in range(sim_object.num_spins):
            results = job.result()        
            avg_mag_qc = []
            temp = []
            i = 1
            print("Post-processing qubit {} data".format(j+1))
            with open(sim_object.namevar,'a') as tempfile:
                tempfile.write("Post-processing qubit {} data\n".format(j+1))
            for c in sim_object.ibm_circuits_list:
                    result_dict = results.get_counts(c)
                    temp.append(system_mag_process(result_dict, sim_object.shots))
                    if i % (sim_object.steps+1) == 0:
                            avg_mag_qc.append(temp)
                            temp = []
                    i += 1
            
            # QC
            if "True" in sim_object.plot_flag:
                fig, ax = plt.subplots()
                plt.plot(range(sim_object.steps+1), avg_mag_qc[0])
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
            sim_object.result_out_list.append(avg_mag_qc[0])
            existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, sim_object.num_spins))
            np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,sim_object.num_spins,len(existing)+1),avg_mag_qc[0])
        sim_object.result_matrix=np.stack(sim_object.result_out_list)           
        print("Done")
        with open(sim_object.namevar,'a') as tempfile:
            tempfile.write("Done\n")		
            
def individual_magnetization(sim_object,result): 
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

	if "QS" in sim_object.QCQS:
		for j in range(sim_object.num_spins):
			avg_mag_sim = []
			temp = []
			i = 1
			print("Post-processing qubit {} data".format(j+1))
			with open(sim_object.namevar,'a') as tempfile:
				tempfile.write("Post-processing qubit {} data\n".format(j+1))
			for c in sim_object.ibm_circuits_list:
				result_dict = result_noise.get_counts(c)
				temp.append(individual_mag_process(result, sim_object.shots,j))
				if i % (sim_object.steps+1) == 0:
					avg_mag_sim.append(temp)
					temp = []
				i += 1

			if "True" in sim_object.plot_flag:
				fig, ax = plt.subplots()
				plt.plot(range(sim_object.steps+1), avg_mag_sim[0])
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
			sim_object.result_out_list.append(avg_mag_sim[0])
			existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, sim_object.num_spins))
			np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,sim_object.num_spins,len(existing)+1),avg_mag_sim[0])
		sim_object.result_matrix=np.stack(sim_object.result_out_list)
		print("Done")
		with open(sim_object.namevar,'a') as tempfile:
			tempfile.write("Done\n")
	elif "QC" in sim_object.QCQS:
        for j in range(sim_object.num_spins):
            results = job.result()        
            avg_mag_qc = []
            temp = []
            i = 1
            print("Post-processing qubit {} data".format(j+1))
            with open(sim_object.namevar,'a') as tempfile:
                tempfile.write("Post-processing qubit {} data\n".format(j+1))
            for c in sim_object.ibm_circuits_list:
                    result_dict = results.get_counts(c)
                    temp.append(individual_mag_process(result_dict, sim_object.shots,j))
                    if i % (sim_object.steps+1) == 0:
                            avg_mag_qc.append(temp)
                            temp = []
                    i += 1
            
            # QC
            if "True" in sim_object.plot_flag:
                fig, ax = plt.subplots()
                plt.plot(range(sim_object.steps+1), avg_mag_qc[0])
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
            sim_object.result_out_list.append(avg_mag_qc[0])
            existing=glob.glob("data/Spin {} Average Magnetization Data, Qubits={}, num_*.txt".format(j+1, sim_object.num_spins))
            np.savetxt("data/Spin {} Average Magnetization Data, Qubits={}, num_{}.txt".format(j+1,sim_object.num_spins,len(existing)+1),avg_mag_qc[0])
        sim_object.result_matrix=np.stack(sim_object.result_out_list)           
        print("Done")
        with open(sim_object.namevar,'a') as tempfile:
            tempfile.write("Done\n")

def energy(directionlist,result, plotchoice):
	#calculations here