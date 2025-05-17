#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

import numpy as np
import yaml
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Base directory of the current script
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Select pricing problem from config
# init_pricing, solve_pricing = get_subproblem(config["subproblem"])
init_pricing, solve_pricing = None, None

# Load data on rank 0
if rank == 0:  
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    item_data = {
                    "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy"))
                }

    agent_data = {
                    "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy"))
                }

    np.random.seed(0)
    errors = np.random.normal(0, 1, size=(config["num_simuls"], 255, 493))

    data = {
                "item_data": item_data,
                "agent_data": agent_data,
                "errors": errors,
                "obs_bundle": obs_bundle
            }
else:
    data = None

### User-defined feature oracle
def get_x_k(self, i_id, B_j, local= False):  
    modular = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    return np.concatenate((modular[B_j].sum(0), [-B_j.sum() **(1.5)]))

### Demand oracle
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

# Create the BundleChoice instance
quadsupermod_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
quadsupermod_demo.scatter_data()
quadsupermod_demo.local_data_to_torch()


# my_test.compute_estimator_row_gen()
print("Rank", rank, "Hi", print(quadsupermod_demo.torch_local_agent_data['modular'].shape))

if rank == 0:
    print(data["agent_data"]["modular"].shape)