import numpy as np
import gurobipy as gp
import os

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")


modular_i_j_k = np.load(os.path.join(INPUT_DIR, "modular.npy"))
obs_bundle = np.load(os.path.join(INPUT_DIR, "obs_bundles.npy"))

num_agents, num_items, num_features = modular_i_j_k.shape

master = gp.Model()
master.setAttr('ModelSense', gp.GRB.MAXIMIZE)
master.setParam('OutputFlag', 0)

lambda_k = master.addVars(num_features, name="lambda_k")
u_i = master.addVars(num_agents, name="u_i")


x_hat_k = np.einsum('ijk,ij->k', modular_i_j_k, obs_bundle)
print("x_hat_k:", x_hat_k)
np.random.seed(0)
error_i_j = np.random.normal(0, 1, (num_agents, num_items))


master.setObjective( gp.quicksum( lambda_k[k] * x_hat_k[k] for k in range(num_features) ) -
                        gp.quicksum( u_i[i] for i in range(num_agents) ) )

master.addConstrs((u_i[i] >= gp.quicksum(modular_i_j_k[i, j, k] * lambda_k[k] for k in range(num_features)) + error_i_j[i, j]
                    for i in range(num_agents) for j in range(num_items)))

master.optimize()

print("parameters:", [lambda_k[k].x for k in range(num_features)])
print("objective value:", master.objVal)
