#!/bin/env python

from bundlechoice import BundleChoice

import numpy as np
import yaml
from mpi4py import MPI
import os
import platform

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


BASE_DIR = os.path.dirname(__file__)
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro") 
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:  
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    num_agents = config["dimensions"]["num_agents"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    num_simulations = config["dimensions"]["num_simulations"]

    item_data = {
        "modular": -np.eye(num_items),
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }
    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
    }

    np.random.seed(1995)
    errors = np.random.normal(0, 1, size=(num_simulations, num_agents, num_items))

    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    input_data = None

# Broadcast dimensions to all ranks
num_features = comm.bcast(num_features if rank == 0 else None, root=0)

# # Run the estimation
combinatorial_auction = BundleChoice()
combinatorial_auction.load_config(CONFIG_PATH)
combinatorial_auction.data.load_and_scatter(input_data)
combinatorial_auction.features.build_from_data()
combinatorial_auction.subproblems.load()
# combinatorial_auction.subproblems.initialize_local()
# combinatorial_auction.subproblems.init_and_solve(np.ones(num_features))
result = combinatorial_auction.row_generation.solve()
if rank == 0:
    print(f"\n{result.summary()}")
