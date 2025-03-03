#!/bin/env python

from mpi4py import MPI
import numpy as np
from itertools import chain
import datetime
from pricing import pricing
from master import master
import gurobipy as gp


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

################################################################
####################### LOAD DATA ##############################
################################################################

quadratic_j_j_k = np.load('./data/quadratic_characteristic_j_j_k.npy')
weight_j = np.load('./data/weight_j.npy')

if rank == 0:
    # Load full data only on Rank 0
    modular_i_j_k = np.load('./data/modular_characteristics_i_j_k.npy')
    capacity_i = np.load('./data/capacity_i.npy')
    epsilon_si_j = np.load('./data/epsilon_si_j.npy')

    num_agents = len(capacity_i)
    num_simulations = int(epsilon_si_j.shape[0] / num_agents)
    num_characteristics =  modular_i_j_k.shape[2] + quadratic_j_j_k.shape[2] 
    
    # Create data chunks to scatter
    
    indeces_chunks = np.array_split(np.kron(np.ones(num_simulations, dtype = int),
                                            np.arange(num_agents)), 
                                            comm_size)

    data_chunks = []
    start = 0  
    for r in range(comm_size):
        end = start + len(indeces_chunks[r]) 
        modular = np.concatenate((epsilon_si_j[start:end,:,None], 
                                    modular_i_j_k[indeces_chunks[r]]), axis = 2)

        data_chunks.append({
                            "indices": indeces_chunks[r],
                            "modular": modular,
                            "capacity": capacity_i[indeces_chunks[r]],
                            })
        start = end 

    del modular_i_j_k, capacity_i, indeces_chunks
else:
    data_chunks = None 

# Scatter the data chunks from Rank 0 to all ranks
local_data = comm.scatter(data_chunks, root=0)

local_indices = local_data["indices"]
local_modular = local_data["modular"]
local_capacity = local_data["capacity"]
del local_data

if rank == 0:
    del data_chunks 
    print("############################################################################################################################")
    print('Data loaded and scattered. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S"))
    print("############################################################################################################################")


solution_master_pb = np.load('output/solution_master_pb.npy')


################################################################
####################### MAIN LOOP ##############################
################################################################

max_iters = 200
if rank == 0:
    model = gp.read("output/master_pb.mps")
    model.read('output/master_pb.bas')

    slack_counter = {constr.ConstrName: 1 if constr.CBasis == 0 else 0 for constr in model.getConstrs()}
    
for iteration in range(max_iters):

    ### Solve pricing
    local_results = [
                    pricing(local_modular[ell], quadratic_j_j_k, weight_j, local_capacity[ell],
                            solution_master_pb)
                    for ell in range(len(local_indices))
                    ]

    # Gather pricing results at rank 0
    pricing_results = comm.gather(local_results, root=0)

    ### Solve master at rank 0  
    if rank == 0:
        print("############################################################################################################################")
        print("############################################################################################################################")
        print("ITERATION: ", iteration)
        print('TIME after pricing: ', datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("############################################################################################################################") 
        print("############################################################################################################################")

        pricing_results  =  np.vstack(list(chain.from_iterable(pricing_results)))    
        np.save(f'output/constraints/pricing_results_{iteration}.npy', pricing_results)

        stop, solution_master_pb  = master(pricing_results, slack_counter)
        

        print('TIME after master: ', datetime.datetime.now().time().strftime("%H:%M:%S"))
    else:
        stop  = None
        solution_master_pb = None

    # Broadcast master results to all ranks
    stop  = comm.bcast(stop , root=0)
    solution_master_pb = comm.bcast(solution_master_pb,root = 0)

    if stop:
        if rank == 0:
            print("############################################################################################################################")
            print("############################################################################################################################")
            print('SOLUTION FOUND:',solution_master_pb[:num_characteristics])
            print("############################################################################################################################")
            print("############################################################################################################################")
        break

    
    