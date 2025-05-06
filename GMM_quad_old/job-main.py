#!/bin/env python

from mpi4py import MPI
import numpy as np
import datetime
import gurobipy as gp
import os

from initialize_master import init_master
from pricing import init_pricing, solve_pricing
from master import solve_master
from utilities import my_print, generate_data_chunks 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()


TOL_CERTIFICATE = 1e-3
MAX_SLACK_COUNTER = 100
TOL_ROW_GENERATION = 0
ROW_GENERATION_DECAY = 0.5
NUM_SIMULATIONS = 1

MAX_ITERS = 20
MIN_ITERS = np.log(TOL_CERTIFICATE / (TOL_ROW_GENERATION - 1)) / np.log(ROW_GENERATION_DECAY)

################################################################################################################################ 
####################### LOAD DATA ##############################################################################################
################################################################################################################################

# Get path to the GMM_quad_old/data folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Load agent-independent data on all ranks
quadratic_j_j_k = np.load(os.path.join(DATA_DIR, 'quadratic_characteristic_j_j_k.npy'))
weight_j = np.load(os.path.join(DATA_DIR, 'weight_j.npy'))

# Get MPI rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Load full individual-specific data on Rank 0 only
    modular_i_j_k = np.load(os.path.join(DATA_DIR, 'modular_characteristics_i_j_k.npy'))
    capacity_i = np.load(os.path.join(DATA_DIR, 'capacity_i.npy'))
    num_agents = len(capacity_i)
    # epsilon_si_j = np.load('GMM_quad_old/data/epsilon_si_j.npy')[:num_agents * NUM_SIMULATIONS]
    np.random.seed(0)
    epsilon_si_j = np.random.normal(0, 1, size=(num_agents * NUM_SIMULATIONS, len(weight_j)))
    
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
local_pricing = [init_pricing(weight_j, local_capacity[ell]) for ell in range(len(local_capacity))] 

if rank == 0:
    # Initialize master problem
    master_pb, vars_tuple, lambda_k_iter, p_j_iter = init_master(epsilon_si_j, modular_i_j_k, quadratic_j_j_k, 
                                                                num_simulations = NUM_SIMULATIONS)
    # Initialize slack counter                                                               
    slack_counter = {"MAX_SLACK_COUNTER": MAX_SLACK_COUNTER}
else:
    lambda_k_iter, p_j_iter = None, None

# Broadcast master solution to all ranks
lambda_k_iter, p_j_iter = comm.bcast((lambda_k_iter, p_j_iter), root=0)

####################### MAIN LOOP ##############################

for iteration in range(MAX_ITERS):
    ### Solve pricing and update problems/new rows
    local_new_rows = np.array([solve_pricing(pricing_pb, local_modular[ell], quadratic_j_j_k, lambda_k_iter, p_j_iter) 
                                for ell, pricing_pb in enumerate(local_pricing)])

    # Gather pricing results at rank 0
    pricing_results = comm.gather(local_new_rows, root= 0)

    if rank == 0:
        my_print([["ITERATION: ",iteration], ['TIME after pricing: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])
        
        ### Solve master at rank 0 
        pricing_results = np.concatenate(pricing_results)
        stop, lambda_k_iter, p_j_iter = solve_master(master_pb, vars_tuple, pricing_results, slack_counter,
                                                    tol_opt = TOL_CERTIFICATE, 
                                                    tol_row_gen = TOL_ROW_GENERATION)
        TOL_ROW_GENERATION *= ROW_GENERATION_DECAY
    else:
        stop, lambda_k_iter, p_j_iter = None, None, None

    # Broadcast master results to all ranks
    stop, lambda_k_iter, p_j_iter = comm.bcast((stop, lambda_k_iter, p_j_iter) , root=0)

    # Break loop if stop is True and min iters is reached
    if stop and iteration > MIN_ITERS:
        if rank == 0:
            my_print([['SOLUTION FOUND:', lambda_k_iter]])
        break

    
    