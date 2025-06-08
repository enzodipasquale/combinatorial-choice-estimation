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
                    "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic.npy"))
                }

    agent_data = {
                    "modular": np.load(os.path.join(INPUT_DIR, "modular.npy"))
                }

    np.random.seed(34254)
    errors = np.random.normal(0, 1, size=(config["num_simuls"], config["num_agents"], config["num_items"]))
    errors *= 10

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
    modular = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    quadratic = self.item_data["quadratic"]
    return np.concatenate((
                            np.einsum('jk,j->k', modular, B_j),
                            np.einsum('jlk,j,l->k', quadratic, B_j, B_j)
                            ))
        
# Demand orable from library
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

# Run estimation
quadsupermod_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
quadsupermod_demo.scatter_data()
quadsupermod_demo.local_data_to_torch()

quadsupermod_demo.compute_estimator_row_gen()


# lambda_k_star = np.ones(config["num_features"]) 
# # lambda_k_star = np.array([3.41268895 ,1.33776912, 0.47635377, 2.83031925, 0.98361723])
# B_si_j = quadsupermod_demo.solve_pricing_offline(lambda_k_star)

# if rank == 0:
#     x_hat_i_k = quadsupermod_demo.get_x_i_k(quadsupermod_demo.obs_bundle)
#     x_hat_k = x_hat_i_k.sum(0)
#     objective = (
#                 x_hat_k @ lambda_k_star -
#                 quadsupermod_demo.get_x_si_k(B_si_j).sum(0) @ lambda_k_star - (quadsupermod_demo.error_si_j* B_si_j).sum()
#                 )
#     print("Objective value:", objective)