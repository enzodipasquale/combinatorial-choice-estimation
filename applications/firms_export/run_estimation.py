#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

import numpy as np
import yaml
from mpi4py import MPI
import os


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:  
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")

    obs_bundle = np.load(os.path.join(INPUT_DIR, "obs_bundles.npy"))

    item_data = {   
                    "modular": np.load(os.path.join(INPUT_DIR, "modular_j_k.npy")),
                    "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_j_j_k.npy"))
                }

    agent_data = {
                    "modular": np.load(os.path.join(INPUT_DIR, "modular_i_j_k.npy"))
                }

    np.random.seed(34254)
    shape = (config["num_simuls"], config["num_agents"], config["num_items"])
    errors = np.random.normal(0, 1, size=shape)

    # # Correlation structure
    # rho_0 = .16
    # rho_d = .82
    # sigmasq = 1
    # distance_j_j = np.load(os.path.join(INPUT_DIR, "distance_j_j.npy"))
    # Covariance = sigmasq * rho_0 * np.exp(- rho_d * distance_j_j)
    # from scipy.linalg import sqrtm
    # cov_sqrt = sqrtm(Covariance)
    # for s in range(config["num_simuls"]):
    #     for i in range(config["num_agents"]):
    #         errors[s, i] = cov_sqrt @ errors[s, i]

    # Blocking shocks
    # p = 0.97
    # random_vals = np.random.rand(*shape)
    # errors = np.where(random_vals < p, errors, - float('inf'))

    print("Check if there is a minus infinity in the errors" , np.any(np.isneginf(errors)))

    data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": errors,
            "obs_bundle": obs_bundle
            }
else:
    data = None

# User-defined feature oracle
def get_x_k(self, i_id, B_j, local= False):
    modular_item = self.item_data["modular"]
    modular_agent = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    quadratic_item = self.item_data["quadratic"]
    return np.concatenate(( np.einsum('jk,j->k', modular_item, B_j),
                            np.einsum('jk,j->k', modular_agent, B_j),
                            np.einsum('jlk,j,l->k', quadratic_item, B_j, B_j)
                            ))
        
# Demand orable from library
init_pricing, solve_pricing = get_subproblem(config["subproblem_name"])

# Run estimation
firms_export = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
firms_export.scatter_data()
firms_export.local_data_to_torch()
lambda_k_star , _ = firms_export.compute_estimator_row_gen()


results = firms_export.solve_pricing_offline(lambda_k_star)
if rank == 0:

    aggregate_demand_obs = obs_bundle.mean(0)
    sorted_indices = np.argsort(aggregate_demand_obs)[::-1]
    predicted_demand = results.mean(0)[sorted_indices]

    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(aggregate_demand_obs[sorted_indices], label='Observed Demand', marker='o')
    plt.plot(predicted_demand, label='Predicted Demand', marker='x')
    plt.title('Observed vs Predicted Demand')
    plt.xlabel('Items (sorted by observed demand)')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, "marginals.png"))
    plt.show()


    import matplotlib.pyplot as plt
    marg_obs = obs_bundle.sum(1)
    marg_pred = results.sum(1)
    plt.hist(marg_obs, bins=50, alpha=0.5, color='blue', label='observed')
    plt.hist(marg_pred , bins=50, alpha=0.5, color='orange', label='predicted')

    plt.yscale('log') 
    plt.xlabel("Number of destinations")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "count_distribution.png"))
    plt.show()




