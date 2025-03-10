#!/bin/env python

import numpy as np
import gurobipy as gp   
from utilities import check_gap

def init_pricing(weight_j, capacity):

    # Create subproblem
    subproblem = gp.Model() 
    subproblem.setParam('OutputFlag', 0)

    # Create variables
    B_j = subproblem.addMVar(len(weight_j), vtype = gp.GRB.BINARY)

    # Knapsack constraint
    subproblem.addConstr(weight_j @ B_j <= capacity)
    subproblem.update()

    return subproblem 


def solve_pricing(subproblem, modular_j_k, quadratic_j_j_k ,lambda_k, p_j):

    ### Define objective from data and master solution 
    num_MOD = modular_j_k.shape[1] - 1
    L_j =  modular_j_k[:,0] + modular_j_k[:,1:] @ lambda_k[:num_MOD] -  p_j
    Q_j_j = quadratic_j_j_k @ lambda_k[num_MOD: ]

    # Set objective
    B_j = subproblem.getVars()
    subproblem.setObjective(B_j @ L_j + B_j @ Q_j_j @ B_j, gp.GRB.MAXIMIZE)

    # Solve the updated subproblem
    subproblem.optimize()
    optimal_bundle = np.array(subproblem.x, dtype=bool)
    value = subproblem.objVal
    check_gap(subproblem.MIPGap)

    ### Compute value of characteristics at optimal bundle (1+1+K+J)
    row =   np.concatenate(([value],
                            (modular_j_k[optimal_bundle]).sum(0), 
                            quadratic_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1)),
                            subproblem.x
                            ))
    return subproblem, row



