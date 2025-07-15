#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

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

    item_data = {
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }

    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
    }

    np.random.seed(1995)
    errors = np.random.normal(0, 1, size=(config["num_simuls"], config["num_agents"], config["num_items"]))

    data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    data = None

# Run the estimation
combinatorial_auction = BundleChoice(config)
combinatorial_auction.load_data(data)
combinatorial_auction.build_feature_oracle_from_data()
combinatorial_auction.scatter_data()
combinatorial_auction.compute_estimator_row_gen()
