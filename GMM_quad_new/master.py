#!/bin/env python

import numpy as np
import gurobipy as gp
from utilities import print_master_info, update_slack_counter, my_print

def solve_master(master_pb, vars_tuple, pricing_results, slack_counter = None, tol_opt = 1e-3, tol_row_gen = 1):

    num_objects = 493

    u_si_star = pricing_results[:,0]
    characteristics_star_si_k = pricing_results[:,1: - num_objects]
    B_star_si_j = pricing_results[:,- num_objects:]

    lambda_k, u_si, p_j = vars_tuple

    # Check certificates
    u_si_master = u_si.x

    max_reduced_cost = max([u_si_star[si] - u_si_master[si] for si in range(len(u_si_star))])
    if max_reduced_cost < tol_opt:
        return True, master_pb, (lambda_k, u_si, p_j), slack_counter

    # Add new constraints
    new_constrs = [si for si in range(len(u_si_star)) if u_si_star[si] > u_si_master[si] * (1+tol_row_gen)]
    master_pb.addConstrs((  u_si[si] + B_star_si_j[si,:] @ p_j >=  
                            characteristics_star_si_k[si,0] + characteristics_star_si_k[si,1:] @ lambda_k 
                        for si in new_constrs
                        ))

    # Solve master problem
    master_pb.optimize()

    # Update slack_counter
    slack_counter, num_constrs_removed = update_slack_counter(master_pb, slack_counter)

    # Save results
    master_pb.write('output/master_pb.mps')
    master_pb.write('output/master_pb.bas')

    # Print some information
    print_master_info(master_pb, u_si_star, u_si_master, max_reduced_cost, lambda_k, num_constrs_removed, len(new_constrs))

    return False, master_pb, (lambda_k, u_si, p_j), slack_counter
    
