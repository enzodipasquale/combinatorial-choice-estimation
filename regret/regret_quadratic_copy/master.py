#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *

def master(pricing_results, tol = 1e-3):
    num_characteristics = 4
    num_agents = 255
    num_objects = 493

    u_star_i = pricing_results[:,0]
    characteristics_star_i_k = pricing_results[:,1: - num_objects]
    B_star_i_j = pricing_results[:,- num_objects:]

    if characteristics_star_i_k.shape[1] != num_characteristics + 1:
        raise ValueError('Characteristics do not match')
    if not np.all(np.logical_or(B_star_i_j == 0, B_star_i_j == 1)):
        raise ValueError('B_star_i_j is not binary')



    # Load master model and solution from previous iteration
    model = read("output/master_pb.mps")

    lambda_k = gp.tupledict({k: model.getVarByName(f"parameters[{k}]") for k in range(num_characteristics)})
    u_i = gp.tupledict({i: model.getVarByName(f"utilities[{i}]") for i in range(num_agents)})
    p_j = gp.tupledict({j: model.getVarByName(f"prices[{j}]") for j in range(num_objects)})

    solution_master_pb = np.load('output/solution_master_pb.npy')
    u_i_master = solution_master_pb[num_characteristics: -num_objects]

    # Check certificates
    max_reduced_cost = np.max(u_star_i - u_i_master)
    if max_reduced_cost < tol:
        np.save('output/SOLUTION_FOUND.npy', True)

    else:
        # Add new constraints
        model.addConstrs((
            u_i[i] + gp.quicksum(B_star_i_j[i,j] * p_j[j] for j in range(num_objects)) >= characteristics_star_i_k[i,0] 
            + gp.quicksum(characteristics_star_i_k[i,k+1] * lambda_k[k] for k in range(num_characteristics))
            for i in range(num_agents) 
            if u_i_master[i] < u_star_i[i]
                        ))
        model.update()
        
        # Compute master solution
        lambda_k.start = solution_master_pb[:num_characteristics]
        p_j.start = solution_master_pb[-num_objects:]
        u_i.start = u_star_i

        model.setParam('OutputFlag', 0)
        model.optimize()
        solution_master_pb = np.array(model.x)

        # Save results
        # num_constraints = len(model.getConstrs())
        np.save('output/solution_master_pb.npy', solution_master_pb)
        model.write('output/master_pb.mps')

        # Print some information
        print('--------------------------------------------------------')
        print('Max reduced cost ', max_reduced_cost)
        print('Max u_star_i     ', np.max(u_star_i))
        print('Max u_i_master   ', np.max(u_i_master))
        print('Max price        ', np.max(solution_master_pb[-num_objects:]))
        print('--------------------------------------------------------')
        print("Constraints added:    ", (u_i_master < u_star_i).sum(), '  ',np.argsort(u_star_i - u_i_master)[::-1][:5] )
        print("Parameters: ", solution_master_pb[:num_characteristics])
        print("Objective:  ", model.objVal)

        # return False
        
      
        
       

