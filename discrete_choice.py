import numpy as np
import gurobipy as gp
import os


num_agents = 50
num_items = 50
num_features = 5
np.random.seed(0)
modular_i_j_k = np.random.normal(0, 1, (num_agents, num_items, num_features))
error_i_j = np.random.normal(0, 1, (num_agents, num_items))
lambda_star = np.ones(num_features)


choice_i = (modular_i_j_k @ lambda_star + error_i_j).argmax(axis=1)
obs_bundle = np.zeros((num_agents, num_items), dtype=bool)
obs_bundle[np.arange(num_agents), choice_i] = True



results = []
for test in range(60):

    num_agents, num_items, num_features = modular_i_j_k.shape
    num_simul = 10

    master = gp.Model()
    master.setAttr('ModelSense', gp.GRB.MAXIMIZE)
    master.setParam('OutputFlag', 0)

    lambda_k = master.addVars(num_features, name="lambda_k")
    u_si = master.addVars(num_simul * num_agents, name="u_i")


    x_hat_k = np.einsum('ijk,ij->k', modular_i_j_k, obs_bundle)
    # print("x_hat_k:", x_hat_k)
    # np.random.seed(test)
    error_si_j = np.random.normal(0, 1, (num_simul * num_agents, num_items))
    master.setObjective( gp.quicksum( num_simul * lambda_k[k] * x_hat_k[k] for k in range(num_features) ) -
                            gp.quicksum( u_si[si] for si in range(num_simul * num_agents) ) )

    master.addConstrs((u_si[si] >= gp.quicksum(modular_i_j_k[si % num_agents, j, k] * lambda_k[k] for k in range(num_features)) + error_si_j[si % num_agents, j]
                        for si in range(num_simul * num_agents) for j in range(num_items)))

    master.optimize()

    estimated_lambda = np.array([lambda_k[k].x for k in range(num_features)])
    # print("parameters:", [lambda_k[k].x for k in range(num_features)])
    # print("objective value:", master.objVal)
    results.append(estimated_lambda)


results = np.array(results)
std_results = np.std(results, axis=0)
print("std_results:", std_results)