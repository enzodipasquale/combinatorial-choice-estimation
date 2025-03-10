
import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *
import datetime
import time

### Load modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j
modular_characteristics_i_j_k = np.load('../data/modular_characteristics_i_j_k.npy')
quadratic_characteristic_j_j_k = np.load('../data/quadratic_characteristic_j_j_k.npy')
matching_i_j = np.load('../data/matching_i_j.npy')
epsilon_si_j = np.load('../data/epsilon_si_j.npy')

num_simulations = int(epsilon_si_j.shape[0] / matching_i_j.shape[0])
epsilon_s_i_j = epsilon_si_j.reshape(num_simulations, matching_i_j.shape[0], matching_i_j.shape[1])

def initialize_pb(modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j):
    print('Time: ', datetime.datetime.now())
    print('#' * 100)

    tic = time.time()
    num_agents, num_objects = matching_i_j.shape
    num_MOD = modular_characteristics_i_j_k.shape[2]
    num_QUAD = quadratic_characteristic_j_j_k.shape[2]
    num_characteristics = num_MOD + num_QUAD

    phi_hat_i_k = np.concatenate(((
                                modular_characteristics_i_j_k * matching_i_j[:, :, None]).sum(1),
                                np.einsum('jlk,ij,il->ik', quadratic_characteristic_j_j_k, matching_i_j, matching_i_j
                                )), axis = 1)
    
    phi_hat_k = phi_hat_i_k.sum(0)


    model = gp.Model('GMM_pb')
    model.setParam('Method', 0)

    # Variables 
    lambda_k = model.addVars(num_characteristics, lb= -1e9, ub = 1e9 , name='parameters')
    u_si = model.addVars(num_simulations * num_agents,  lb= 0, ub = 1e20 , name='utilities')
    p_j = model.addVars(num_objects,  lb= 0, ub = 1e20  , name='prices')

    # Objective
    model.setObjective( gp.quicksum(phi_hat_k[k]  * lambda_k[k] for k in range(num_characteristics)) 
                        - (1/ num_simulations) * u_si.sum() - p_j.sum(), sense=gp.GRB.MAXIMIZE)

    # Non negativity constraint lambda_k[2]>=0
    for k in range(num_MOD, num_characteristics):
        model.addConstr(lambda_k[k] >= 0, name=f"non_negativity_lambda_{k}")

    # Constraints
    # model.addConstrs((
    #         u_si[si] + gp.quicksum(matching_i_j[si % num_agents,j] * p_j[j] for j in range(num_objects)) 
    #         >= epsilon_si_j[si,matching_i_j[si % num_agents]].sum() 
    #         + gp.quicksum(phi_hat_i_k[si % num_agents, k]  * (1+1e-9)* lambda_k[k] for k in range(num_characteristics))
    #         for si in range(num_simulations * num_agents)
    #                 ))

    phi_i_all_k = np.concatenate((modular_characteristics_i_j_k.sum(1), 
                                 np.tile(quadratic_characteristic_j_j_k.sum((0,1)), (num_agents,1)) ), 
                                 axis = 1)
    model.addConstrs((
            u_si[si] + gp.quicksum(p_j[j] for j in range(num_objects)) 
            >= epsilon_si_j[si].sum() 
            + gp.quicksum(phi_i_all_k[si % num_agents, k]  * (1+1e-9)* lambda_k[k] for k in range(num_characteristics))
            for si in range(num_simulations * num_agents)
                    ))

    # Solve master problem
    model.optimize()

    UB = - np.einsum('sij,ij->',epsilon_s_i_j, matching_i_j)/num_simulations
    total_time = time.time() - tic
    print_init_master(model, num_characteristics, num_simulations, num_agents, num_objects, phi_hat_k, total_time, UB)

    ### Save master problem
    model.write('../output/master_pb.mps')
    model.write('../output/master_pb.bas')

    return model , np.array(model.x)




def print_init_master(model, num_characteristics, num_simulations, num_agents, num_objects, phi_hat_k, time, UB):
    print('#' * 100)
    print('num_characteristics: ', num_characteristics)
    print('num_simulations: ', num_simulations)
    print('num_agents: ', num_agents)
    print('num_objects: ', num_objects)
    print('phi_hat: ', phi_hat_k)
    print('-'* 100)
    print('lenght solution: ', len(model.x))
    print('K+SI+J:', num_characteristics + num_simulations * num_agents + num_objects)
    print('Number of constraints: ', len(model.getConstrs()))
    print('-'* 100)
    print('Value of objective function: ', model.objVal)
    print('upper bound: ' , UB)
    print('min solution:', np.array(model.x).min(), 'max solution:', np.array(model.x).max())
    print('-'* 100)
    print(np.array(model.x)[:num_characteristics])
    print('-'* 100)
    print('Time: ', time)


model, solution_master_pb = initialize_pb(modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j)  