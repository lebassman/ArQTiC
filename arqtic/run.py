from H_simulator import *

#create the instance of the Heisenberg class with the default "input_file.txt" input file found in the directory
inst=Heisenberg()

#connect to the IBM backend
inst.connect_account()

#generate the circuits
inst.generate_circuits()

#return the circuits to a list you can work with
circ_list=inst.return_circuits()


#optional if you want to run the circuits
# inst.run_circuits()  
