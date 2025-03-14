#!/bin/env python

import numpy as np
import gurobipy as gp
import datetime
from utilities import print_init_master, my_print


def init_master(epsilon_si_j, modular_i_j_k, quadratic_j_j_k, num_simulations):

    ### Load modular_i_j_k, quadratic_j_j_k, matching_i_j
    matching_i_j = np.load('./data/matching_i_j.npy')
    num_agents, num_objects = matching_i_j.shape
    num_characteristics = modular_i_j_k.shape[2] + quadratic_j_j_k.shape[2]

    num_MOD = modular_i_j_k.shape[2]
    num_QUAD = quadratic_j_j_k.shape[2]
    num_characteristics = num_MOD + num_QUAD

    phi_hat_i_k = np.concatenate((
                        np.einsum('ijk,ij->ik', modular_i_j_k, matching_i_j),
                        np.einsum('jlk,ij,il->ik', quadratic_j_j_k, matching_i_j, matching_i_j)
                        ), axis = 1)
    
    phi_hat_k = phi_hat_i_k.sum(0)

    master_pb = gp.Model('GMM_pb')
    master_pb.setParam('Method', 0)
    master_pb.setParam('LPWarmStart', 2)
    
    # Variables and Objective
    master_pb.setAttr('ModelSense', gp.GRB.MAXIMIZE)
    lambda_k = master_pb.addMVar(num_characteristics, obj = phi_hat_k, lb= -1e9, ub = 1e9 , name='parameter')
    u_si = master_pb.addMVar(num_simulations * num_agents, obj = - (1/ num_simulations), name='utility')
    p_j = master_pb.addMVar(num_objects, obj = -1 , name='price')

    # Non negativity constraint lambda_k[2]>=0
    for k in range(num_MOD, num_characteristics):
        master_pb.addConstr(lambda_k[k] >= 0, name=f"non_negativity_lambda_{k}")

    # Constraints
    phi_i_all_k = np.concatenate((modular_i_j_k.sum(1), 
                                 np.tile(quadratic_j_j_k.sum((0,1)), (num_agents,1)) ), 
                                 axis = 1)
    master_pb.addConstrs((
            u_si[si] + p_j.sum() >= epsilon_si_j[si].sum() + phi_i_all_k[si % num_agents, :] @ lambda_k
            for si in range(num_simulations * num_agents)
                        ))

    # Solve master problem
    master_pb.optimize()

    epsilon_s_i_j = epsilon_si_j.reshape(num_simulations, num_agents, num_objects)
    UB = - np.einsum('sij,ij->',epsilon_s_i_j, matching_i_j)/num_simulations

    # Print some information
    print_init_master(master_pb, num_MOD, num_QUAD, num_simulations, num_agents, num_objects, phi_hat_k, UB)
    my_print([['Pricing and Master init done. Time: ', datetime.datetime.now().time().strftime("%H:%M:%S")]])

    return master_pb, (lambda_k, u_si, p_j), lambda_k.x, p_j.x

