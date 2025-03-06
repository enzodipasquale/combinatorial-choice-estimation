#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *
import time

def pricing(modular_j_k, quadratic_j_j_k, weight_j, capacity, solution_master_pb, OutputFlag = False):
    tic = time.time()
    num_MOD = modular_j_k.shape[1] - 1
    num_characteristics = num_MOD +  quadratic_j_j_k.shape[2]
    num_objects = len(weight_j)

    ### Load master solution
    # solution_master_pb =  np.load('output/solution_master_pb.npy')
    lambda_k = solution_master_pb[: num_characteristics]
    p_j = solution_master_pb[-num_objects:]

    ### Define objective from data and master solution 
    value_j =  modular_j_k[:,0] + modular_j_k[:,1:] @ lambda_k[:num_MOD] -  p_j
                
    Q_j_j = quadratic_j_j_k @ lambda_k[num_MOD: ]

    ### Solve Pricing Problem (Quadratic Knapsack Problem)
    env = gp.Env(empty=True)
    if not OutputFlag:
        env.setParam("OutputFlag",0)
    env.start()
    model = gp.Model(env = env) 

    B_j = model.addVars(num_objects, vtype=GRB.BINARY)
    linear_obj = gp.quicksum(value_j[j] * B_j[j] for j in range(num_objects))
    quadratic_obj = gp.quicksum(Q_j_j[j, k] * B_j[j] * B_j[k] for j in range(num_objects) for k in range(num_objects))
    model.setObjective( linear_obj + quadratic_obj, GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weight_j[j] * B_j[j] for j in range(num_objects)) <= capacity)

    model.optimize()
    optimal_bundle = np.array([B_j[j].x for j in range(num_objects)], dtype=bool)
    value = model.objVal
    gap = model.MIPGap
    env.dispose()
    if gap > 1e-2:
        print("--------------------------------------------------------------------------------------------------------------------------------")
        print('Warning: Gap is too high', gap)
        print("--------------------------------------------------------------------------------------------------------------------------------")
        
    ### Compute value of characteristics at optimal bundle

    row =       np.concatenate(([value],
                                (modular_j_k[optimal_bundle]).sum(0), 
                                quadratic_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1)),
                                optimal_bundle.astype(float)
                                ))

   
    # Save results
    return row

