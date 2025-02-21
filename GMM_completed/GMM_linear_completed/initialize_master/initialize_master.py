
import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *


### Load modular_characteristics_i_j_k, matching_i_j
modular_characteristics_i_j_k = np.load('../data/modular_characteristics_i_j_k.npy')
matching_i_j = np.load('../data/matching_i_j.npy')

epsilon_si_j = np.load('../data/epsilon_si_j.npy')

num_simulations = int(epsilon_si_j.shape[0] / matching_i_j.shape[0])


def initialize_pb(modular_characteristics_i_j_k, matching_i_j):

    num_agents, num_objects = matching_i_j.shape
    num_characteristics = modular_characteristics_i_j_k.shape[2]

    phi_hat_i_k = (modular_characteristics_i_j_k * matching_i_j[:, :, None]).sum(1)
    phi_hat_k = phi_hat_i_k.sum(0)

    model = gp.Model('regret_pb')

    # Variables 
    lambda_k = model.addVars(num_characteristics, lb= -1e9, ub = 1e9 , name='parameters')
    u_si = model.addVars(num_simulations * num_agents,  lb= 0, ub = 1e9 , name='utilities')
    p_j = model.addVars(num_objects,  lb= 0, ub = 1e9  , name='prices')


    # Objective
    model.setObjective( gp.quicksum(phi_hat_k[k]*lambda_k[k] for k in range(num_characteristics)) 
                        - (1/ num_simulations) * u_si.sum() - p_j.sum(), sense=gp.GRB.MAXIMIZE)

    # Constraints

    model.addConstrs((
            u_si[si] + gp.quicksum(matching_i_j[si % num_agents,j] * p_j[j] for j in range(num_objects)) 
            >= epsilon_si_j[si,matching_i_j[si % num_agents]].sum() 
            + gp.quicksum(phi_hat_i_k[si % num_agents,k] * lambda_k[k] for k in range(num_characteristics))
            for si in range(num_simulations * num_agents)
                    ))

                        
    # Solve master problem
    model.setParam('OutputFlag', 0)
    model.optimize()


    print('num_characteristics: ', num_characteristics)
    print('num_simulations: ', num_simulations)
    print('num_agents: ', num_agents)
    print('num_objects: ', num_objects)
    print('phi_hat: ', phi_hat_k)

    print('lenght solution: ', len(model.x))
    print('K+SI+J:', num_characteristics + num_simulations * num_agents + num_objects)
    print('Number of constraints: ', len(model.getConstrs()))

    print('min solution:', np.array(model.x).min(), 'max solution:', np.array(model.x).max())

    print(np.array(model.x)[:num_characteristics])

    return model , np.array(model.x)


### Save master problem
model, solution_master_pb = initialize_pb(modular_characteristics_i_j_k, matching_i_j)    

# np.save('../output/solution_master_pb.npy', solution_master_pb)
# model.write('../output/master_pb.mps')


