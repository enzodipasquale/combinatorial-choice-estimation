#!/bin/env python

import numpy as np
import gurobipy as gp
from utilities import print_master_info, update_slack_counter

def solve_master(master_pb, solution_master_pb, pricing_results, slack_counter = None , tol_opt = 1e-3 , tol_row_generation = 1):

    num_objects = 493
    num_simulated_agents = len(pricing_results)

    u_si_star = pricing_results[:,0]
    characteristics_star_si_k = pricing_results[:,1: - num_objects]
    B_star_si_j = pricing_results[:,- num_objects:]

    num_characteristics = characteristics_star_si_k.shape[1] -1

    all_vars = master_pb.getVars()
    lambda_k, u_si, p_j = all_vars[:num_characteristics], all_vars[num_characteristics: -num_objects] , all_vars[-num_objects:]

    # Check certificates
    u_si_master = solution_master_pb[num_characteristics: -num_objects]
    max_reduced_cost = np.max(u_si_star - u_si_master)
    if max_reduced_cost < tol_opt:
        return True, solution_master_pb, slack_counter

    # Add new constraints
    master_pb.addConstrs((
        u_si[si] + gp.quicksum(B_star_si_j[si,j] * p_j[j] for j in range(num_objects)) >= characteristics_star_si_k[si,0] 
        + gp.quicksum(characteristics_star_si_k[si,k+1] * lambda_k[k] for k in range(num_characteristics))
        for si in range(num_simulated_agents) 
        if  u_si_star[si] > u_si_master[si] * (1+tol_row_generation)
                    ))

    # Solve master problem
    master_pb.optimize()
    solution_master_pb = np.array(master_pb.x)

    # Save results
    master_pb.write('output/master_pb.mps')
    master_pb.write('output/master_pb.bas')
    
    # Update slack_counter
    slack_counter, constraints_removed = update_slack_counter(master_pb, slack_counter)

    # Print some information
    print_master_info(master_pb, u_si_star , u_si_master, num_characteristics ,num_objects , constraints_removed, tol_row_generation)

    return False, solution_master_pb, slack_counter
    
