#!/bin/env python

import numpy as np
import gurobipy as gp
from utilities import print_master_info, update_slack_counter, my_print

def solve_master(master_pb, vars_tuple, pricing_results, slack_counter = None, tol_opt = 1e-3, tol_row_gen = 0):

    lambda_k, u_si, p_j = vars_tuple
    num_objects = p_j.size

    u_si_star = pricing_results[:,0]
    eps_si_star = pricing_results[:,1]
    X_star_si_k = pricing_results[:,2: - num_objects]
    B_star_si_j = pricing_results[:,- num_objects:]

    # Check certificates
    u_si_master = u_si.x
    max_reduced_cost = np.max(u_si_star - u_si_master)
    if max_reduced_cost < tol_opt:
        return True, lambda_k.x, p_j.x

    # Add new constraints
    new_constrs_id = np.where(u_si_star > u_si_master * (1+tol_row_gen))[0]
    master_pb.addConstrs((  
                        u_si[si] + B_star_si_j[si,:] @ p_j >= eps_si_star[si] + X_star_si_k[si] @ lambda_k 
                        for si in new_constrs_id
                        ))

    # Update slack_counter
    slack_counter, num_constrs_removed = update_slack_counter(master_pb, slack_counter)

    # Solve master problem
    master_pb.optimize()

    # Save results
    # master_pb.write('output/master_pb.mps')
    # master_pb.write('output/master_pb.bas')

    # Print some information
    print_master_info(u_si_star, u_si_master, lambda_k, num_constrs_removed, len(new_constrs_id))
                        
    return False, lambda_k.x, p_j.x
    
