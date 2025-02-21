#!/bin/env python

from mpi4py import MPI
import numpy as np
import pickle  # For flexible data serialization
from pricing import pricing
from master import master
import datetime
from itertools import chain

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

### Load data and send to each rank

quadratic_characteristics_j_j_k = np.load('./data/quadratic_characteristic_j_j_k.npy')
weight_j = np.load('./data/weight_j.npy')

# Only Rank 0 will load the full dataset and prepare the chunks
if rank == 0:
    # Load full data ONCE on Rank 0
    modular_characteristics_i_j_k = np.load('./data/modular_characteristics_i_j_k.npy')
    capacity_i = np.load('./data/capacity_i.npy')
    num_agents = len(capacity_i)
    
    # Create chunks for scattering
    indices_chunks = [np.arange(r, num_agents, comm_size) for r in range(comm_size)]
    modular_chunks = [modular_characteristics_i_j_k[indices] for indices in indices_chunks]
    capacity_chunks = [capacity_i[indices] for indices in indices_chunks]
    
    # Package data for each rank as dictionaries
    data_chunks = [
        {
            "indices": indices_chunks[r],
            "modular": modular_chunks[r],
            "capacity": capacity_chunks[r]
        }
        for r in range(comm_size)
    ]
    
    del modular_characteristics_i_j_k, capacity_i

else:
    # Initialize placeholders for received data
    data_chunks = None

# Scatter the data chunks from Rank 0 to all ranks
local_data = comm.scatter(data_chunks, root=0)

# Unpack received data
local_indices = local_data["indices"]
local_modular = local_data["modular"]
local_capacity = local_data["capacity"]

del local_data

# Now, each rank has its own portion of the data


comm.Barrier()


### Run loop 

for iteration in range(600):
    # Solve pricing
    local_results = []
    for i, i_id in enumerate(local_indices):
        results_pricing = pricing(local_modular[i], quadratic_characteristics_j_j_k, weight_j, local_capacity[i])
        local_results.append([i_id,results_pricing])

    # Gather results at rank 0
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
       
        pricing_results = sorted(chain.from_iterable(gathered_results), key=lambda x: x[0])
        # print i_id for each element in pricing_results
        # print([(result[0], result[1][3] )for result in pricing_results])

        # Convert to NumPy arrays
        length = len(pricing_results)
        pricing_results = np.array([result[1] for result in pricing_results], dtype=object)
        u_star_i = np.array(pricing_results[:, 0])
        characteristics_star_i_k = np.vstack(pricing_results[:, 1])
        B_star_i_j = np.vstack(pricing_results[:, 2])   
        del gathered_results, pricing_results
        
        print("##############################################################")
        print('TIME  after gathering   : ', datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("ITERATION: ", iteration, "      length: ", length)
        print("##############################################################") 
        if length != num_agents:
            raise ValueError("Some agents did not return results")
            
        # Solve master at rank 0                                    
        master(u_star_i, characteristics_star_i_k, B_star_i_j)
 
    comm.Barrier()
    if rank == 0:
        print('TIME  after master   : ', datetime.datetime.now().time().strftime("%H:%M:%S"))

    # if np.load('output/SOLUTION_FOUND.npy'):
    #     break
    
    