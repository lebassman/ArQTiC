#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:55:45 2020

@author: lindsaybassman
"""
import numpy as np
from arqtic.program import Program, random_bitstring
from arqtic.hamiltonians import Ising_Hamiltonian
from arqtic.arqtic_for_ibm import run_ibm
import qiskit as qk
from qiskit import Aer, IBMQ, execute


#define system variables
N = 2 #number of qubits
Jz = 0.1 #ising interaction strength
mu_x = 0.3 #transverse magnetic field strength
param_free_ham = Ising_Hamiltonian(2, Jz, [mu_x, 0, 0]) #parameter-free Hamiltonian

#define simulation variables
beta = 1.0 #inverse temperature of systems
dbeta = 0.1 #step-size in beta for QITE
tau = 10 #total trajectory time to evolve lambda from 0 to 1
dtau = 2.0 #time-step for trajectory
num_steps = int(tau/dtau)
T = 2 #total number of trajectories
dt =  dtau #timestep for Trotter approximation: setting equal to dtau means one trotter-step per time-step in evolution
lamba_protocol = np.linspace((dtau/tau),1.0,num_steps)
    

#interfacing with IBM
#load account
#qk.IBMQ.load_account()
#see available backends
#provider = qk.IBMQ.get_provider(group='open')
#print(provider.backends)
#set up simulator
simulator = Aer.get_backend('qasm_simulator')


#all programs run will be a combintaion of three sub-programs:
# 1.prog_ips to create the intial product state
# 2. prog_qmetts to evolve to the intial thermal state
# 3. prog_hamEvol to evolve the system to a given time-step of the trajectory

## Create list of programs to be run
programs = []

#create prog_qmetts
prog_qmetts = Program(N)
ham0 = Ising_Hamiltonian(N, Jz, [0,0,0])
#prog_qmetts.make_qmetts_prog(beta, dbeta, ham0)

#get list of measured_metts_states from which to start each trajectory
measured_metts_state = []
#first state should be random
bitstring = random_bitstring(N)
measured_metts_state.append(bitstring)

#subsequent entries are derived from running QMETTS on the previously derived
#state and measuring a random observable to get the state for the subsequent run
for i in range(T):
    prog = Program(N)
    prog.make_ps_prog(measured_metts_state[i])
    prog.append_program(prog_qmetts)
    #make random measurement operator
    rand_meas_prog = Program(N)
    rand_meas_prog.make_rand_measurement_prog()
    prog.append_program(rand_meas_prog)
    measured_metts_state.append(run_ibm(simulator, prog))


#loop over trajectories
for i in range(T):
    #make the initial product state program
    prog_ips = Program(N)
    prog_ips.make_ps_prog(measured_metts_state[i])
    #loop over time-steps in trajectory i    
    for step in range(num_steps):
        #make Hamilton Evolution program for given time-step of given trajectory
        prog_hamEvol = Program(N)
        prog_hamEvol.make_hamEvol_prog(step, dtau, dt, lamba_protocol, param_free_ham)
        #build main program combing IPS preparation, QMETTS evolution, and Hamiltonian evolution
        prog_main= Program(N)
        prog_main.append_program(prog_ips)
        prog_main.append_program(prog_qmetts)
        prog_main.append_program(prog_hamEvol)
        programs.append(prog_main)
        


