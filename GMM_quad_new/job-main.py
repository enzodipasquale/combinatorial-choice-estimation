#!/bin/env python

from mpi4py import MPI
import numpy as np
from itertools import chain
import datetime
import gurobipy as gp


from initialize_master import init_master
from pricing import init_pricing, solve_pricing
from master import solve_master
from utilities import my_print, generate_data_chunks 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()


TOL_CERTIFICATE = 1e-3
MAX_SLACK_COUNTER = 6
TOL_ROW_GENERATION = 4
ROW_GENERATION_DECAY = 0.5
NUM_SIMULATIONS = 50

# (TOL_ROW_GENERATION - 1) * (ROW_GENERATION_DECAY ** MIN_ITERS) < TOL_CERTIFICATE
MIN_ITERS = np.log(TOL_CERTIFICATE / (TOL_ROW_GENERATION - 1)) / np.log(ROW_GENERATION_DECAY)

MAX_ITERS = 100

################################################################################################################################ 
####################### LOAD DATA ##############################################################################################
################################################################################################################################

# Load agent-independent data on all ranks
quadratic_j_j_k = np.load('./data/quadratic_characteristic_j_j_k.npy')
weight_j = np.load('./data/weight_j.npy')

if rank == 0:
    # Load full individual specific data on Rank 0 only
    modular_i_j_k = np.load('./data/modular_characteristics_i_j_k.npy')
    capacity_i = np.load('./data/capacity_i.npy')
    num_agents = len(capacity_i)
    epsilon_si_j = np.load('./data/epsilon_si_j.npy')[:num_agents * NUM_SIMULATIONS]
    
    # Create data chunks 
    data_chunks = generate_data_chunks(epsilon_si_j , modular_i_j_k , capacity_i , comm_size)
else:
    data_chunks = None 

# Scatter data chunks from Rank 0 to all ranks
local_data = comm.scatter(data_chunks, root=0)

local_modular = local_data["modular"]
local_capacity = local_data["capacity"]

if rank == 0:
    my_print([['Data loaded and scattered. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])
    

################################################################################################################################
####################### ROW GENERATION #########################################################################################
################################################################################################################################

################# Initialize pricing and master ################

# Initialize pricing 
local_pricing = [ init_pricing(weight_j, local_capacity[0]) for ell in range(len(local_capacity)) ] 

if rank == 0:
    # Initialize master problem
    master_pb , vars_tuple , slack_counter = init_master(epsilon_si_j, modular_i_j_k, quadratic_j_j_k,
                                                        num_simulations = NUM_SIMULATIONS)
    slack_counter["MAX_SLACK_COUNTER"] = MAX_SLACK_COUNTER
    
    lambda_k_iter = vars_tuple[0].x
    p_j_iter = vars_tuple[2].x
else:
    lambda_k_iter, p_j_iter = None, None

# Broadcast master solution to all ranks
lambda_k_iter = comm.bcast(lambda_k_iter, root=0)
p_j_iter = comm.bcast(p_j_iter, root=0)


####################### MAIN LOOP ##############################

for iteration in range(MAX_ITERS):
    ### Solve pricing and update problems/new rows
    local_new_rows = []
    for index, pricing_pb in enumerate(local_pricing):
        local_pricing[index], new_rows = solve_pricing( pricing_pb, 
                                                        local_modular[index], quadratic_j_j_k, 
                                                        lambda_k_iter, p_j_iter)
        local_new_rows.append(new_rows)
    
    # Gather pricing results at rank 0
    pricing_results = comm.gather(local_new_rows, root=0)

    ### Solve master at rank 0  
    if rank == 0:
        my_print([["ITERATION: ",iteration], ['TIME after pricing: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])

        pricing_results  =  np.vstack(list(chain.from_iterable(pricing_results))) 
        stop, master_pb, vars_tuple, slack_counter = solve_master(master_pb, vars_tuple, pricing_results, 
                                                                slack_counter,
                                                                tol_opt = TOL_CERTIFICATE, 
                                                                tol_row_gen = TOL_ROW_GENERATION)
        
        lambda_k_iter = vars_tuple[0].x
        p_j_iter = vars_tuple[2].x

        TOL_ROW_GENERATION *= ROW_GENERATION_DECAY
    else:
        stop, lambda_k_iter, p_j_iter = None, None, None

    # Broadcast master results to all ranks
    stop  = comm.bcast(stop , root=0)
    lambda_k_iter = comm.bcast(lambda_k_iter,root = 0)
    p_j_iter = comm.bcast(p_j_iter, root=0)

    # Break loop if stop is True
    if stop and iteration > MIN_ITERS:
        if rank == 0:
            my_print([['SOLUTION FOUND:', lambda_k_iter]])
        break

    
    