
import numpy as np
from gurobipy import GRB
import gurobipy as gp
from gurobipy import *


### Load modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j
modular_characteristics_i_j_k = np.load('../data/modular_characteristics_i_j_k.npy')
quadratic_characteristic_j_j_k = np.load('../data/quadratic_characteristic_j_j_k.npy')
matching_i_j = np.load('../data/matching_i_j.npy')


def initialize_pb(modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j):

    num_agents, num_objects = matching_i_j.shape
    num_characteristics = modular_characteristics_i_j_k.shape[2] + quadratic_characteristic_j_j_k.shape[2]

    phi_hat = np.concatenate(((modular_characteristics_i_j_k * matching_i_j[:, :, None]).sum((0,1)),
                np.einsum('jlk,ij,il->k', quadratic_characteristic_j_j_k, matching_i_j, matching_i_j)))
    

    model = gp.Model('regret_pb')

    # Variables (MINMAXREGRET)
    lambda_k = model.addVars(num_characteristics - 1, lb= -1e9, ub = 1e9 , name='parameters')
    u_i = model.addVars(num_agents,  lb= 0, ub = 1e9 , name='utilities')
    p_j = model.addVars(num_objects,  lb= 0, ub = 1e9  , name='prices')

    # Objective
    model.setObjective( gp.quicksum(phi_hat[k+1]*lambda_k[k] for k in range(num_characteristics -1)) 
                        - u_i.sum() - p_j.sum(), sense=gp.GRB.MAXIMIZE)
    # Solve master problem
    model.setParam('OutputFlag', 0)
    model.optimize()


    print('num_characteristics: ', num_characteristics)
    print('num_agents: ', num_agents)
    print('num_objects: ', num_objects)
    print('phi_hat: ', phi_hat)

    print('lenght solution: ', len(model.x))
    print('K-1+I+J:', num_characteristics - 1 + num_agents + num_objects)
    print('Number of constraints: ', len(model.getConstrs()))

    return model , np.array(model.x)


### Save master problem
model, solution_master_pb = initialize_pb(modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j)    



# np.save('../output/solution_master_pb.npy', solution_master_pb)
# model.write('../output/master_pb.mps')
