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
        
# Pricing subproblem from bundlechoice library
init_pricing, solve_pricing = get_subproblem(config["subproblem"])


quadsupermod_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
quadsupermod_demo.scatter_data()
quadsupermod_demo.local_data_to_torch()

quadsupermod_demo.compute_estimator_row_gen()
