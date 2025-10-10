#!/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpi4py import MPI

from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
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
    num_agents = config["dimensions"]["num_agents"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    num_simuls = config["dimensions"]["num_simuls"]
    shape = (num_simuls, num_agents, num_items)
    errors = np.random.normal(0, 1, size=shape)

    input_data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": errors,
            "obs_bundle": obs_bundle
            }
else:
    input_data = None


# # Run the estimation
firms_export = BundleChoice()
firms_export.load_config(CONFIG_PATH)
firms_export.data.load_and_scatter(input_data)
firms_export.features.build_from_data()
firms_export.subproblems.load()
theta_hat = firms_export.row_generation.solve()

results = firms_export.subproblems.init_and_solve(theta_hat)
if rank == 0:

    aggregate_demand_obs = obs_bundle.mean(0)
    sorted_indices = np.argsort(aggregate_demand_obs)[::-1]
    predicted_demand = results.mean(0)[sorted_indices]

    # Plotting the results
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




