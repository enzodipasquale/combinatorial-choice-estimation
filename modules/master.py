#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *

num_characteristics = 3
num_agents = 255
num_objects = 493

tol = 1e-3

# Load master solution from previous iteration
model = read("output/master_pb.mps")

lambda_k = gp.tupledict({k: model.getVarByName(f"parameters[{k}]") for k in range(num_characteristics)})
u_i = gp.tupledict({i: model.getVarByName(f"utilities[{i}]") for i in range(num_agents)})
p_j = gp.tupledict({j: model.getVarByName(f"prices[{j}]") for j in range(num_objects)})

solution_master_pb = np.load('output/solution_master_pb.npy')
u_i_master = solution_master_pb[num_characteristics: -num_objects]

### Load pricing solution 
new_constraints_dict = {}
u_star_i = np.zeros(num_agents)
for i in range(num_agents):
    u_star_i[i] = np.load(f'new_constraints/u_star_{i}.npy')
    optimal_bundle = np.load(f'new_constraints/optimal_bundle_{i}.npy')
    characteristics_star = np.load(f'new_constraints/characteristics_star_{i}.npy')
    characteristic_0_star = np.load(f'new_constraints/characteristic_0_star_{i}.npy')
    if u_i_master[i] < u_star_i[i]:
        new_constraints_dict[i] = (
                                u_i[i] + gp.quicksum(p_j[j] for j in np.where(optimal_bundle)[0]) >= characteristic_0_star + 
                                gp.quicksum( (characteristics_star[k]) * lambda_k[k] for k in range(num_characteristics))
                                    )
### Check certificates
max_reduced_cost  = np.max(u_star_i - u_i_master )
print("###############################")
print('Max reduced cost ',max_reduced_cost)
print('Max u_star_i     ',np.max(u_star_i))
print('Mean u_star_i    ',np.mean(u_star_i))
print('Max u_i_master   ',np.max(u_i_master))
print('Mean u_i_master',np.mean(u_i_master))
print('#############')

if max_reduced_cost < tol:
    np.save('output/SOLUTION_FOUND.npy', True)

else:
    ### Add new constraints
    model.addConstrs((constraint for constraint in new_constraints_dict.values()))
    model.update()
    print("Number of constraints", len(model.getConstrs()))
    print("Constraints added:   ",len(new_constraints_dict))

    ### Update master solution
    lambda_k.start = solution_master_pb[ : num_characteristics]
    p_j.start = solution_master_pb[ - num_objects : ]
    u_i.start = u_star_i
    
    model.setParam('OutputFlag', 0)
    model.optimize()
    solution_master_pb = np.array(model.x)
    print("Paramters: ",solution_master_pb[:num_characteristics])
    print("Objective: ",model.objVal)
    print("###############################")
    
    # Save the solution
    np.save('output/solution_master_pb.npy', solution_master_pb)
    model.write('output/master_pb.mps')
