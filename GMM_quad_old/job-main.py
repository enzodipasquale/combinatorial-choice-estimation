#!/bin/env python

from mpi4py import MPI
import numpy as np
from itertools import chain
import datetime
import gurobipy as gp

from pricing import init_pricing, solve_pricing
from master import solve_master
from utilities import my_print, generate_data_chunks #, check_gap, print_master_info

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

if rank == 0:
    my_print([['Started job. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])

    TOL_CERTIFICATE = 1e-3
    MAX_SLACK_COUNTER = 6
    TOL_ROW_GENERATION = 4
    ROW_GENERATION_DECAY = 0.5

    # (TOL_ROW_GENERATION - 1) * (ROW_GENERATION_DECAY ** MIN_ITERS) < 1e-6
    MIN_ITERS = np.ceil(np.log(1e-6 / (TOL_ROW_GENERATION - 1)) / np.log(ROW_GENERATION_DECAY))

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
    epsilon_si_j = np.load('./data/epsilon_si_j.npy')
    
    # Create data chunks 
    data_chunks = generate_data_chunks(epsilon_si_j , modular_i_j_k , capacity_i , comm_size)
else:
    data_chunks = None 

# Scatter data chunks from Rank 0 to all ranks
rank_data = comm.scatter(data_chunks, root=0)

rank_modular = rank_data["modular"]
rank_capacity = rank_data["capacity"]
rank_tasks = len(rank_capacity)

if rank == 0:
    my_print([['Data loaded and scattered. Time: ', datetime.datetime.now().time().strftime("%d-%m-%Y %H:%M:%S")]])
    

################################################################################################################################
####################### ROW GENERATION #########################################################################################
################################################################################################################################

################# Initialize pricing and master ################

# Initialize pricing 
rank_pricing = {}
rank_pricing["pricing_pb"] = [ init_pricing(weight_j, rank_capacity[ell]) for ell in range(rank_tasks) ] 
    
if rank == 0:
    # Initialize master problem
    master_pb = gp.read("output/master_pb.mps")
    master_pb.read('output/master_pb.bas')
    master_pb.setParam('LPWarmStart',2)
    master_pb.optimize()
    solution_master_pb = np.array(master_pb.x)

    # Initialize slack counter
    slack_counter = {constr.ConstrName: 1 if constr.Slack > 0 else 0 for constr in master_pb.getConstrs()}
    slack_counter["MAX_SLACK_COUNTER"] = MAX_SLACK_COUNTER

    my_print([['Pricing and Master init done. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])
else:
    solution_master_pb = None

# Broadcast master solution to all ranks
solution_master_pb = comm.bcast(solution_master_pb, root=0)


####################### MAIN LOOP ##############################


for iteration in range(MAX_ITERS):

    ### Solve pricing and update problems/new rows
    rank_pricing["pricing_pb"], rank_pricing["new_row"]  = zip(*[
                                                solve_pricing(pricing_pb, rank_modular[ell], quadratic_j_j_k, solution_master_pb)
                                                for ell, pricing_pb in enumerate(rank_pricing["pricing_pb"])
                                                ])
    # Gather pricing results at rank 0
    pricing_results = comm.gather(rank_pricing["new_row"] , root=0)

    ### Solve master at rank 0  
    if rank == 0:
        my_print([["ITERATION: ",iteration], 
                ['TIME after pricing: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])

        pricing_results  =  np.vstack(list(chain.from_iterable(pricing_results)))   
        stop, solution_master_pb, slack_counter  = solve_master(master_pb, pricing_results, slack_counter, 
                                                                tol_opt = TOL_CERTIFICATE, 
                                                                tol_row_generation = TOL_ROW_GENERATION)
        
        TOL_ROW_GENERATION *= ROW_GENERATION_DECAY

        my_print([['TIME after master: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])
    else:
        stop, solution_master_pb = None, None

    # Broadcast master results to all ranks
    stop  = comm.bcast(stop , root=0)
    solution_master_pb = comm.bcast(solution_master_pb,root = 0)

    # Break loop if stop is True
    if stop and iteration >= MIN_ITERS:
        if rank == 0:
            num_characteristics =  modular_i_j_k.shape[2] + quadratic_j_j_k.shape[2] 
            my_print([['SOLUTION FOUND:',solution_master_pb[:num_characteristics]]])
        break

    
    