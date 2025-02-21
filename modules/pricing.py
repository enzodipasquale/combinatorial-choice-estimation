#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
import time
import sys

num_characteristics = 3
num_agents = 255
num_objects = 493

MOD_characteristics = 2
SMOD_characteristics = 1

# lambda_k_MOD = np.array([18, 3,4])
# lambda_SMOD = 15

### Load master solution
solution_master_pb =  np.load('output/solution_master_pb.npy')
lambda_k = solution_master_pb[: num_characteristics]
lambda_k_MOD = np.concatenate(([1],lambda_k[:MOD_characteristics]))
lambda_SMOD = lambda_k[-SMOD_characteristics:]


### Load Data
quadratic_characteristic_j_j = np.load('data/quadratic_characteristic_j_j_k.npy')[0]
Q_j_j = quadratic_characteristic_j_j * lambda_SMOD 
weight_j = np.load('data/weight_j.npy')
modular_characteristics_i_j_k = np.load('data/modular_characteristics_i_j_k.npy')
profit_i_j = modular_characteristics_i_j_k @ lambda_k_MOD
capacity_i = np.load('data/capacity_i.npy')

# Find id of the array job
id = int(sys.argv[1]) -1
profit_j = profit_i_j[id] - solution_master_pb[ -num_objects:]
capacity = capacity_i[id]

### Solve Pricing Problem
model = gp.Model() 
model.setParam('OutputFlag', 0)

B_i = model.addVars(num_objects, vtype=GRB.BINARY)

# Objective
linear_obj = gp.quicksum(profit_j[j] * B_i[j] for j in range(num_objects))
quadratic_obj = gp.quicksum(Q_j_j[j, k] * B_i[j] * B_i[k] for j in range(num_objects) for k in range(num_objects))
model.setObjective( linear_obj + quadratic_obj, GRB.MAXIMIZE)

# Constraint
model.addConstr(gp.quicksum(weight_j[j] * B_i[j] for j in range(num_objects)) <= capacity)

model.optimize()
optimal_bundle = np.array([B_i[j].x for j in range(num_objects)], dtype=bool)
u_star_i = model.objVal  

### Compute new constraints (minmax regret)

characteristic_0_star = (modular_characteristics_i_j_k[id, :, 0] * optimal_bundle).sum()

characteristics_star = np.append(
                        (modular_characteristics_i_j_k[id, :, 1:] * optimal_bundle[:,None]).sum(axis=0), 
                        quadratic_characteristic_j_j[np.ix_(optimal_bundle, optimal_bundle)].sum()
                        )

# Save results
np.save(f'new_constraints/u_star_{id}.npy', u_star_i)
np.save(f'new_constraints/characteristic_0_star_{id}.npy', characteristic_0_star)
np.save(f'new_constraints/characteristics_star_{id}.npy', characteristics_star)
np.save(f'new_constraints/optimal_bundle_{id}.npy', optimal_bundle)
