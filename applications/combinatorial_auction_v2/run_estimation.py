#!/bin/env python
"""
Main estimation script for combinatorial auction v2.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
    num_agents = None
    num_items = None
    num_features = None
    num_simulations = None

# Broadcast dimensions to all ranks
num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
num_simulations = comm.bcast(num_simulations, root=0)

# Run the estimation
combinatorial_auction = BundleChoice()
combinatorial_auction.load_config(CONFIG_PATH)
combinatorial_auction.data.load_and_scatter(input_data)
combinatorial_auction.features.build_from_data()
combinatorial_auction.subproblems.load()

result = combinatorial_auction.row_generation.solve()

if rank == 0:
    print(f"\n{result.summary()}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save theta_hat
    np.save(os.path.join(OUTPUT_DIR, "theta.npy"), result.theta_hat)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
