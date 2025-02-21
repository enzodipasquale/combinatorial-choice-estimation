#!/bin/env python

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *

def master(pricing_results, tol = 1e-3):

    num_characteristics = 5
    num_simulations = 100
    num_agents = 255
    num_objects = 493

    u_star_si = pricing_results[:,0]
    characteristics_star_si_k = pricing_results[:,1: - num_objects]
    B_star_si_j = pricing_results[:,- num_objects:]

    if characteristics_star_si_k.shape[1] != num_characteristics + 1 or characteristics_star_si_k.shape[0] != num_simulations * num_agents:
        raise ValueError('Characteristics do not match')
    if not np.all(np.logical_or(B_star_si_j == 0, B_star_si_j == 1)):
        raise ValueError('B_star_si_j is not binary')



    # Load master model and solution from previous iteration
    model = read("output/master_pb.mps")
    model.setParam('Threads', 1)

    lambda_k = gp.tupledict({k: model.getVarByName(f"parameters[{k}]") for k in range(num_characteristics)})
    u_si = gp.tupledict({si: model.getVarByName(f"utilities[{si}]") for si in range(num_simulations * num_agents)})
    p_j = gp.tupledict({j: model.getVarByName(f"prices[{j}]") for j in range(num_objects)})

    solution_master_pb = np.load('output/solution_master_pb.npy')
    u_si_master = solution_master_pb[num_characteristics: -num_objects]

    # Check certificates
    max_reduced_cost = np.max(u_star_si - u_si_master)
    if max_reduced_cost < tol:
        return True

    else:
        # Add new constraints
        model.addConstrs((
            u_si[si] + gp.quicksum(B_star_si_j[si,j] * p_j[j] for j in range(num_objects)) >= characteristics_star_si_k[si,0] 
            + gp.quicksum(characteristics_star_si_k[si,k+1] * lambda_k[k] for k in range(num_characteristics))
            for si in range(num_simulations * num_agents) 
            if u_si_master[si] < u_star_si[si]
                        ))
        model.update()
        
        # Compute master solution
        lambda_k.start = solution_master_pb[:num_characteristics]
        p_j.start = solution_master_pb[-num_objects:]
        u_si.start = u_star_si

        model.setParam('OutputFlag', 0)
        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            print(f"Optimization Status: {model.status}")

        solution_master_pb = np.array(model.x)
        dual_solution_master_pb = np.array(model.Pi)

        # Save results
        # num_constraints = len(model.getConstrs())
        np.save('output/solution_master_pb.npy', solution_master_pb)
        np.save('output/dual_solution_master_pb.npy', dual_solution_master_pb)
        model.write('output/master_pb.mps')

        # Print some information
        print('--------------------------------------------------------')
        print('Max reduced cost ', max_reduced_cost)
        print('Max u_star_si     ', np.max(u_star_si))
        print('Max u_si_master   ', np.max(u_si_master))
        print('Max price        ', np.max(solution_master_pb[-num_objects:]))
        print('--------------------------------------------------------')
        print("Constraints added:    ", (u_si_master < u_star_si).sum())#, '  ',np.argsort(u_star_si - u_si_master)[::-1][:5] )
        print("Parameters: ", solution_master_pb[:num_characteristics])
        print("Objective:  ", model.objVal)

        return False
        
      
        
       

