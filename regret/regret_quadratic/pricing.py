#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *

def pricing(modular_characteristics_j_k, quadratic_characteristics_j_j_k, weight_j, capacity):

    num_mod_characteristics = modular_characteristics_j_k.shape[1] - 1
    num_characteristics = num_mod_characteristics +  quadratic_characteristics_j_j_k.shape[2]
    num_objects = len(weight_j)

    ### Load master solution
    solution_master_pb =  np.load('output/solution_master_pb.npy')
    lambda_k = solution_master_pb[: num_characteristics]
    p_j = solution_master_pb[-num_objects:]

    ### Define objective from data and master solutions 
    Q_j_j = quadratic_characteristics_j_j_k @ lambda_k[num_mod_characteristics: ]
    value_j = (modular_characteristics_j_k[:,1:] @ lambda_k[:num_mod_characteristics] 
                + modular_characteristics_j_k[:,0] -  p_j)
   
    ### Solve Pricing Problem (Quadratic Knapsack Problem)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = gp.Model(env = env) 

    B_i = model.addVars(num_objects, vtype=GRB.BINARY)
    linear_obj = gp.quicksum(value_j[j] * B_i[j] for j in range(num_objects))
    quadratic_obj = gp.quicksum(Q_j_j[j, k] * B_i[j] * B_i[k] for j in range(num_objects) for k in range(num_objects))
    model.setObjective( linear_obj + quadratic_obj, GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weight_j[j] * B_i[j] for j in range(num_objects)) <= capacity)

    model.optimize()
    optimal_bundle = np.array([B_i[j].x for j in range(num_objects)], dtype=bool)
    value = model.objVal
    gap = model.MIPGap
    env.dispose()
    if gap > 1e-2:
        print('Warning: Gap is too high', gap)

    ### Compute value of characteristics at optimal bundle
    characteristics_star = np.concatenate((
                                (modular_characteristics_j_k[optimal_bundle]).sum(0), 
                                quadratic_characteristics_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1))
                                ))

    # Save results
    return value, characteristics_star, optimal_bundle, gap

