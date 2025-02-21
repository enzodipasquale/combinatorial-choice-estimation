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

if rank == 0:
    # Load full data ONCE on Rank 0
    modular_characteristics_i_j_k = np.load('./data/modular_characteristics_i_j_k.npy')
    capacity_i = np.load('./data/capacity_i.npy')
    num_agents = len(capacity_i)
    
    # Send data to each rank
    for r in range(1, comm_size):
        subset = np.arange(r, num_agents, comm_size)
        data_to_send = {
            "indices": subset,
            "modular": modular_characteristics_i_j_k[subset],
            "capacity": capacity_i[subset]
                        }
        comm.send(pickle.dumps(data_to_send), dest=r, tag=42)

    # Keep Rank 0's own portion
    local_indices = np.arange(rank, num_agents, comm_size)
    local_modular = modular_characteristics_i_j_k[local_indices]
    local_capacity = capacity_i[local_indices]

    del modular_characteristics_i_j_k, capacity_i, data_to_send


else:
    # Receive data
    received_data = pickle.loads(comm.recv(source=0, tag=42))
    local_indices = received_data["indices"]
    local_modular = received_data["modular"]
    local_capacity = received_data["capacity"]
    
    del received_data

comm.Barrier()
### Run loop 

for iteration in range(1000):

    # Solve pricing
    local_results = []
    for i, i_id in enumerate(local_indices):
        results_pricing = pricing(local_modular[i], quadratic_characteristics_j_j_k, weight_j, local_capacity[i])
        local_results.append([i_id,results_pricing])

    # Gather results at rank 0
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        pricing_results = sorted(chain.from_iterable(gathered_results), key=lambda x: x[0])

        # Convert to NumPy arrays
        length = len(pricing_results)
        pricing_results = np.array([result[1] for result in pricing_results], dtype=object)
        u_star_i = np.array(pricing_results[:, 0])
        characteristics_star_i_k = np.vstack(pricing_results[:, 1])
        B_star_i_j = np.vstack(pricing_results[:, 2])   
        del gathered_results, pricing_results

        print("##############################################################")
        print("ITERATION: ", iteration)
        print('TIME     : ', datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("##############################################################") 
        if length != num_agents:
            raise ValueError("Some agents did not return results")
            
        # Solve master at rank 0                                      
        master(u_star_i, characteristics_star_i_k, B_star_i_j)
    
    comm.Barrier()
    # if np.load('output/SOLUTION_FOUND.npy'):
    #     break
    
    