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


    p = 0.66

    random_vals = np.random.rand(*shape)
    blocking_shocks = np.where(random_vals < p, 0.0,-1e20)
    errors += blocking_shocks

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
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

# Run estimation
firms_export = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
firms_export.scatter_data()
firms_export.local_data_to_torch()

# print(firms_export.obs_bundle.shape)

firms_export.compute_estimator_row_gen()
# Solution found: [1.18849331e+02 0.00000000e+00 4.05564043e-02 2.21206752e-02
#  9.22114324e-02]
# lambda_0: 118.84933133551252
# lambda_1: 0.0
# lambda_2: 0.04055640432761665
# lambda_3: 0.02212067523270796
# lambda_4: 0.09221143236550569