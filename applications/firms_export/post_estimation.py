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


    # p = 0.9
    # random_vals = np.random.rand(*shape)
    # max_val = ((item_data["quadratic"] @ np.ones(2) * 200).sum() + 200* agent_data["modular"].sum((1,2))).max()
    # print("max_val:", max_val)
    # errors = np.where(random_vals < p, errors, - float('inf'))

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
    modular_agent = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    modular_item = self.item_data["modular"]
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

# lambda_1 = 1.059610474320977
# lambda_2 = 0.10969382797095822
# lambda_3 = 0.10037003252260504
# lambda_4 = 1.5651111150566515
# lambda_5 = -0.8273214069366847
# lambda_6 = 0.5552463235956646

lambda_k_star = np.array([1.059610474320977, 0.10969382797095822, 0.10037003252260504,
                            1.5651111150566515, -0.8273214069366847, 0.5552463235956646])

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
    plt.savefig(os.path.join(BASE_DIR, "demand_comparison.png"))
    plt.show()

    

