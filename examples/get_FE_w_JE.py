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
import arqtic.QITE as qite
import qiskit as qk
from qiskit import Aer, IBMQ, execute


#get amouot of work from QC results
def get_work(results, shots):
    work = 0
    for result in results:
        state_vec = result[0]
        count = result[1]
        work_comp = 0
        for i in range(len(state_vec)):
            if (state_vec[i] == 0):
                work_comp += 1
            else:
                work_comp += -1
        work_comp *= count
        work += work_comp
    work = work/shots
    return work

#define system variables
N = 5 #number of qubits
Jz = 0.1 #ising interaction strength
mu_x = 0.3 #transverse magnetic field strength
param_free_ham = Ising_Hamiltonian(N, Jz, [mu_x, 0, 0]) #parameter-free Hamiltonian
ising_ham0 = Ising_Hamiltonian(N, Jz, [0.01, 0, 0]) #Hamiltonian at beginning of parameter sweep

#define simulation variables
tau = 10 #total trajectory time to evolve lambda from 0 to 1
dtau = 1.0 #time-step for trajectory
num_steps = int(tau/dtau)
T = 10 #total number of trajectories
dt =  dtau #timestep for Trotter approximation: setting equal to dtau means one trotter-step per time-step in evolution
lambda_protocol = np.linspace(1.0, 0, num_steps)
dldt = (lambda_protocol[1]-lambda_protocol[0])/dtau # d(lambda)/d(tau)
shots = 1000

#define QITE variables
beta = 2.0 #inverse temperature of systems
dbeta = 0.2 #step-size in beta for QITE
domain = 2 #domain of opertators for QITE


#interfacing with IBM
#load account
#qk.IBMQ.load_account()
#see available backends
#provider = qk.IBMQ.get_provider(group='open')
#print(provider.backends)
#set up simulator
simulator = Aer.get_backend('qasm_simulator')

#for each trajectory two main programs must be run
#the first is for the QMETTS algorithm to generate the initial state for the next trajectory
#this program, prog_qmetts comprises two (alternatingly three) sub-programs:
# 1. prog_ips to create the intial product state
# 2. prog_qite to evolve to the intial thermal state
# alternately apply (3. prog_xBasis to move to x-basis to prepare for measurement
#the second is for the Hamiltonian evolution algorithm to get measurement fors the JE
#this program, prog_JE comprises four sub-programs:
# 1. prog_ips to create the intial product state
# 2. prog_qite to evolve to the intial thermal state
# 3. prog_hamEvol to evolve under system hamiltonian to each time step
# 4. prog_xBasis to move to x-basis to prepare for measurement


#create program to move to x-basis for measurement
prog_xBasis = Program(N)
prog_xBasis.get_x_basis_prog()

#first state should be random
measured_metts_state = random_bitstring(N)
#subsequent entries are derived from running QMETTS on the previously derived
#state and measuring a random observable to get the state for the subsequent run

#need to sum work over each trajectory and then average over works
work = []
#loop over trajectories
for i in range(T):
    print(i)
    psi0 = qite.get_state_from_string(measured_metts_state)
    prog = Program(N)
    prog.make_ps_prog(measured_metts_state)
    #print(measured_metts_state)
    #prog.print_list()
    prog_qite = Program(N)
    #note QITE algorithm should evolve state by beta/2 for temperature beta
    prog_qite.make_QITE_prog(ising_ham0, beta/2.0, dbeta, domain, np.asarray(psi0), 1.0)
    prog.append_program(prog_qite)
    #make and run qmetts program
    prog_qmetts = prog
    #make random measurement operator
    if (i%2 == 0):
        prog_qmetts.append_program(prog_xBasis)
    results = run_ibm(simulator, prog_qmetts,1)
    #update measured metts state for next trajectory
    measured_metts_state = results[0][0]
    #make and run JE program
    prog_JE = prog
    #loop over time-steps in trajectory i
    work_i = 0    
    for step in range(num_steps):
        #print(step)
        #make Hamilton Evolution program for given time-step of given trajectory
        prog_hamEvol = Program(N)
        prog_hamEvol.make_hamEvol_prog(step, dtau, dt, lambda_protocol, param_free_ham)
        #complete JE program: combing IPS preparation, QITE, and Hamiltonian evolution
        prog_JE.append_program(prog_hamEvol)
        prog_JE.append_program(prog_xBasis)
        results = run_ibm(simulator, prog_JE, shots)
        #print(results)
        #print(get_work(results, shots))
        work_i += dldt*dtau*get_work(results, shots)
        work_i *= mu_x
    work.append(work_i)


print(work)
#avg_work = get_average work(work)
#FE = get_free energy(avg_work)
