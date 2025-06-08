import numpy as np
import gurobipy as gp
import os


num_agents = 500
num_items = 5
num_features = 3
np.random.seed(247)
modular_i_j_k = np.random.normal(0, 1, (num_agents, num_items, num_features)) 
# modular_i_j_k = np.repeat(np.random.normal(0, 1, (num_items, num_features))[None,:,:], num_agents, axis=0)

# error_i_j =  np.random.gumbel(0, 1, (num_agents, num_items))
error_i_j =  np.random.normal(0, 1, (num_agents, num_items))

lambda_star = np.ones(num_features)
choice_i = (modular_i_j_k @ lambda_star + error_i_j).argmax(axis=1)
obs_bundle = np.zeros((num_agents, num_items), dtype=bool)
obs_bundle[np.arange(num_agents), choice_i] = True

# x_hat_k = np.einsum('ijk,ij->k', modular_i_j_k, obs_bundle)

results = []

for test in range(50):

    num_agents, num_items, num_features = modular_i_j_k.shape
    num_simul = 1

    master = gp.Model()
    master.setAttr('ModelSense', gp.GRB.MAXIMIZE)
    master.setParam('OutputFlag', 0)
    master.setParam('Method', 1)

    sample_b = np.random.choice(num_agents, num_agents, replace=True)
    modular_b = modular_i_j_k[sample_b, :, :]
    obs_bundle_b = obs_bundle[sample_b, :]
    x_hat_k = np.einsum('ijk,ij->k', modular_b, obs_bundle_b)


    lambda_k = master.addVars(num_features, name="lambda_k", obj = num_simul * x_hat_k)
    u_s_i = master.addVars(range(num_simul), range(num_agents), name="u_s_i", obj = -1, lb = -gp.GRB.INFINITY)
    error_s_i_j =  np.random.normal(0, 1, (num_simul, num_agents, num_items))
    master.addConstrs((u_s_i[s, i] >= gp.quicksum(modular_b[i, j, k] * lambda_k[k] for k in range(num_features)) + error_s_i_j[s, i, j]
                        for s in range(num_simul) for i in range(num_agents) for j in range(num_items)))

    master.optimize()
    if master.status != gp.GRB.OPTIMAL:
        print("No optimal solution found.")
        print("Status:", master.status)
        continue

    estimated_lambda = np.array([lambda_k[k].x for k in range(num_features)])
    # print("Estimated parameters:", estimated_lambda)
    # print("objective value:", master.objVal)
    results.append(estimated_lambda)

results = np.array(results)
print("Estimated parameters (mean):", np.mean(results, axis=0))
print("std_results:", np.std(results, axis=0))