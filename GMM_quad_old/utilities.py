#!/bin/env python

import numpy as np

def generate_data_chunks(epsilon_si_j, modular_i_j_k, capacity_i, comm_size):

    num_agents = len(capacity_i)
    num_simulations = int(epsilon_si_j.shape[0] / num_agents)
    
    # Create data chunks to scatter
    
    i_chunks = np.array_split(np.tile(np.arange(num_agents), num_simulations), comm_size)
    si_chunks = np.array_split(np.arange(num_simulations * num_agents), comm_size)

    data_chunks = [{
                    "modular": np.concatenate((epsilon_si_j[si_chunks[r],:,None], modular_i_j_k[i_chunks[r]]), axis = 2),
                    "capacity": capacity_i[i_chunks[r]],
                    }
                    for r in range(comm_size)]

    return data_chunks


def update_slack_counter(master_pb, slack_counter):
    num_constrs_removed = 0
    for constr in master_pb.getConstrs():
        if constr.ConstrName not in slack_counter:
            slack_counter[constr.ConstrName] = 0
        if constr.Slack > 0:
            slack_counter[constr.ConstrName] += 1
        if slack_counter[constr.ConstrName] >= slack_counter["MAX_SLACK_COUNTER"]:
            master_pb.remove(constr)
            slack_counter.pop(constr.ConstrName)
            num_constrs_removed += 1

    return slack_counter, num_constrs_removed




############## print functions ####################

def check_gap(gap):
    if gap > 1e-2:
        print('X' * 200)
        print('WARNING: Gap is too high', gap, capacity, modular_j_k)
        print('X' * 200)



def  print_master_info(model , u_si_star , u_si_master, lambda_k , p_j, num_constrs_removed, num_constrs_added):

        solution_master_pb = np.array(model.x)
        print('--------------------------------------------------------')
        print('Max reduced cost ', np.max(u_si_star - u_si_master))
        print('Max u_si_star     ', np.max(u_si_star))
        print('Max u_si_master   ', np.max(u_si_master))
        print('Max price        ', max(p_j))
        print('--------------------------------------------------------')
        print("Constraints added:    ",  (u_si_star > u_si_master * (1+tol_row_generation)).sum(), 'out of', (u_si_star > u_si_master ).sum())
        print('--------------------------------------------------------')
        print("Parameters: ", lambda_k)
        print("Objective:  ", model.objVal)
        print('--------------------------------------------------------')
        print("Constraints removed:  ", num_constrs_removed)
        print('--------------------------------------------------------')






def print_init_master(model, num_characteristics, num_simulations, num_agents, num_objects, phi_hat_k, UB):
    print('#' * 100)
    print('num_characteristics: ', num_characteristics)
    print('num_simulations:     ', num_simulations)
    print('num_agents:          ', num_agents)
    print('num_objects:         ', num_objects)
    print('phi_hat:             ', phi_hat_k)
    print('-'* 100)
    print('ObjVal:              ', model.objVal)
    print('upper bound:         ' , UB)
    print('min solution:', np.array(model.x).min(), 'max solution:', np.array(model.x).max())
    print('-'* 100)
    print('parameters:' ,np.array(model.x)[:num_characteristics])
    print('-'* 100)


def my_print(list):
    print("#" * 100)
    print("#" * 100)
    for element in list:
        print(element[0], element[1])
    print("#" * 100)
    print("#" * 100)