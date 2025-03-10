#!/bin/env python

import numpy as np
import gurobipy as gp   
from utilities import check_gap

def init_pricing(weight_j, capacity):
  
    num_objects = len(weight_j)

    model = gp.Model() 
    model.setParam('OutputFlag', 0)

    # Create variables
    B_j = model.addVars(num_objects, vtype= gp.GRB.BINARY)

    # Knapsack constraint
    model.addConstr(gp.quicksum(weight_j[j] * B_j[j] for j in range(num_objects)) <= capacity)
    model.update()

    return model 


def solve_pricing(model, modular_j_k, quadratic_j_j_k ,solution_master_pb):
    num_MOD = modular_j_k.shape[1] - 1
    num_QUAD =quadratic_j_j_k.shape[2]
    num_objects = len(modular_j_k)

    ### Define objective from data and master solution 
    lambda_k = solution_master_pb[: num_MOD + num_QUAD]
    p_j = solution_master_pb[-num_objects:]
    L_j =  modular_j_k[:,0] + modular_j_k[:,1:] @ lambda_k[:num_MOD] -  p_j
    Q_j_j = quadratic_j_j_k @ lambda_k[num_MOD: ]

    # Set objective
    B_j = model.getVars()
    linear_obj = gp.quicksum(L_j[j] * B_j[j] for j in range(num_objects))
    quadratic_obj = gp.quicksum(Q_j_j[j, k] * B_j[j] * B_j[k] for j in range(num_objects) for k in range(num_objects))
    model.setObjective( linear_obj + quadratic_obj, gp.GRB.MAXIMIZE)

    # Solve the updated model
    model.optimize()
    optimal_bundle = np.array([B_j[j].x for j in range(num_objects)], dtype=bool)
    value = model.objVal
    check_gap(model.MIPGap)

    ### Compute value of characteristics at optimal bundle
    row =   np.concatenate(([value],
                            (modular_j_k[optimal_bundle]).sum(0), 
                            quadratic_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1)),
                            optimal_bundle.astype(float)
                            ))

    return model, row



