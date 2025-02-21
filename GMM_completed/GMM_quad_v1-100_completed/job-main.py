#!/bin/env python

from mpi4py import MPI
import numpy as np
from itertools import chain
import datetime
from pricing import pricing
from master import master

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

####################### Load shared data

quadratic_j_j_k = np.load('./data/quadratic_characteristic_j_j_k.npy')
weight_j = np.load('./data/weight_j.npy')

if rank == 0:
    # Load full data only on Rank 0
    modular_i_j_k = np.load('./data/modular_characteristics_i_j_k.npy')
    capacity_i = np.load('./data/capacity_i.npy')
    epsilon_si_j = np.load('./data/epsilon_si_j.npy')

    num_agents = len(capacity_i)
    num_simulations = int(epsilon_si_j.shape[0] / num_agents)

    # Create data chunks to scatter
    
    indeces_chunks = np.array_split(np.kron(np.ones(num_simulations, dtype = int),
                                         np.arange(num_agents)), comm_size)

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
    print("##############################################################")
    print('Data loaded and scattered. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S"))
    print("##############################################################")



####################### Run loop 

for iteration in range(2):
    ### Solve pricing
    local_results = [
                    pricing(local_modular[ell], quadratic_j_j_k, weight_j, local_capacity[ell])
                    for ell in range(len(local_indices))
                    ]

    # Gather results at rank 0
    pricing_results = comm.gather(local_results, root=0)
    comm.barrier()

    ### Solve master at rank 0  
    if rank == 0:
        print("##############################################################")
        print("ITERATION: ", iteration)
        print('TIME after pricing: ', datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("##############################################################") 

        pricing_results  =  np.vstack(list(chain.from_iterable(pricing_results)))                                
        result_master = master(pricing_results)
        print('TIME after master: ', datetime.datetime.now().time().strftime("%H:%M:%S"))

    # Synchronize all ranks
    else:
        result_master = None
    
    # Broadcast the result_master to all ranks
    result_master = comm.bcast(result_master, root=0)
    comm.barrier()

    if result_master == True:
        break

    
    