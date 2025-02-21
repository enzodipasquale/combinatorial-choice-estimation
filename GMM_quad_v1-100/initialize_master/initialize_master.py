
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

    tic = time.time()
    num_agents, num_objects = matching_i_j.shape
    num_characteristics = modular_characteristics_i_j_k.shape[2] + quadratic_characteristic_j_j_k.shape[2]


    phi_hat_i_k = np.concatenate(((
                                modular_characteristics_i_j_k * matching_i_j[:, :, None]).sum(1),
                                np.einsum('jlk,ij,il->ik', quadratic_characteristic_j_j_k, matching_i_j, matching_i_j
                                )), axis = 1)
    
    phi_hat_k = phi_hat_i_k.sum(0) * (1-1e-9)


    model = gp.Model('GMM_pb')
    # model.setParam('OutputFlag', 1)
    model.setParam('Method', 0)
    # Variables (GMM)
    lambda_k = model.addVars(num_characteristics, lb= -1e9, ub = 1e9 , name='parameters')
    u_si = model.addVars(num_simulations * num_agents,  lb= 0, ub = 1e20 , name='utilities')
    p_j = model.addVars(num_objects,  lb= 0, ub = 1e20  , name='prices')


    # Objective
    model.setObjective( gp.quicksum(phi_hat_k[k]  * lambda_k[k] for k in range(num_characteristics)) 
                        - (1/ num_simulations) * u_si.sum() - p_j.sum(), sense=gp.GRB.MAXIMIZE)


    # Non negativity constraint lambda_k[2]>=0
    for k in range(modular_characteristics_i_j_k.shape[2], num_characteristics):
        model.addConstr(lambda_k[k] >= 0, name=f"non_negativity_lambda_{k}")



    

    # Constraints
    model.addConstrs((
            u_si[si] + gp.quicksum(matching_i_j[si % num_agents,j] * p_j[j] for j in range(num_objects)) 
            >= epsilon_si_j[si,matching_i_j[si % num_agents]].sum() 
            + gp.quicksum(phi_hat_i_k[si % num_agents, k]  * (1+1e-9)* lambda_k[k] for k in range(num_characteristics))
            for si in range(num_simulations * num_agents)
                    ))

    model.update()

    # lambda_k.VBasis = [-1] * quadratic_characteristic_j_j_k.shape[2] + [0] * modular_characteristics_i_j_k.shape[2]
    # u_si.VBasis = [0] * num_simulations * num_agents
    # p_j.VBasis = [-1] * num_objects

    # for constr in model.getConstrs():
    #     constr.CBasis = -1
    lambda_k.PStart = np.zeros(num_characteristics)
    u_si.PStart = np.maximum(np.einsum('sij,ij->si',epsilon_s_i_j, matching_i_j),0).flatten()
    p_j.PStart = np.zeros(num_objects)
    print(np.maximum(np.einsum('sij,ij->si',epsilon_s_i_j, matching_i_j),0).flatten().shape)

    for constr in model.getConstrs():
        constr.DStart = 0


    # Solve master problem
    model.optimize()

    #print time
    print('##############################################################')
    print('Time: ', datetime.datetime.now())
    print('##############################################################')

    print('num_characteristics: ', num_characteristics)
    print('num_simulations: ', num_simulations)
    print('num_agents: ', num_agents)
    print('num_objects: ', num_objects)
    print('phi_hat: ', phi_hat_k)
    print('--------------------------------------------------------------------------------------------------------')
    print('lenght solution: ', len(model.x))
    print('K+SI+J:', num_characteristics + num_simulations * num_agents + num_objects)
    print('Number of constraints: ', len(model.getConstrs()))
    print('--------------------------------------------------------------------------------------------------------')
    print('Value of objective function: ', model.objVal)
    print('upper bound: ',- np.einsum('sij,ij->',epsilon_s_i_j, matching_i_j)/num_simulations)
    print('min solution:', np.array(model.x).min(), 'max solution:', np.array(model.x).max())
    print('--------------------------------------------------------------------------------------------------------')
    print(np.array(model.x)[:num_characteristics])
    print('--------------------------------------------------------------------------------------------------------')
    toc = time.time()
    print('Time: ', toc - tic)
    ### Save master problem
    model.write('../output/master_pb.mps')
    model.write('../output/master_pb.bas')
    np.save('../output/solution_master_pb.npy', np.array(model.x))
    np.save('../output/dual_solution_master_pb.npy',np.array(model.pi))

    return model , np.array(model.x)


model, solution_master_pb = initialize_pb(modular_characteristics_i_j_k, quadratic_characteristic_j_j_k, matching_i_j)    





