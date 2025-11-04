#!/bin/env python
"""
Extract estimated parameters from real data estimation.

This script loads the estimated parameters from the combinatorial auction
application and saves them for use in validation experiments.
"""

import numpy as np
import yaml
import os
import sys
from pathlib import Path
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = Path(__file__).parent
AUCTION_DIR = BASE_DIR.parent / "applications" / "combinatorial_auction"
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = AUCTION_DIR / ("config_local.yaml" if IS_LOCAL else "config.yaml")

if rank == 0:
    print(f"Loading estimated parameters from: {AUCTION_DIR}")
    print(f"Config: {CONFIG_PATH}")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:
    INPUT_DIR = AUCTION_DIR / "input_data"
    obs_bundle = np.load(INPUT_DIR / "matching_i_j.npy")

    num_agents = config["dimensions"]["num_agents"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    num_simuls = config["dimensions"]["num_simuls"]

    item_data = {
        "modular": -np.eye(num_items),
        "quadratic": np.load(INPUT_DIR / "quadratic_characteristic_j_j_k.npy"),
        "weights": np.load(INPUT_DIR / "weight_j.npy")
    }
    agent_data = {
        "modular": np.load(INPUT_DIR / "modular_characteristics_i_j_k.npy"),
        "capacity": np.load(INPUT_DIR / "capacity_i.npy"),
    }

    np.random.seed(1995)
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    input_data = None

# Broadcast dimensions
num_features = comm.bcast(num_features if rank == 0 else None, root=0)

# Run estimation to get theta_hat
if rank == 0:
    print("Running estimation on real data...")

combinatorial_auction = BundleChoice()
combinatorial_auction.load_config(str(CONFIG_PATH))
combinatorial_auction.data.load_and_scatter(input_data)
combinatorial_auction.features.build_from_data()
combinatorial_auction.subproblems.load()

theta_hat = combinatorial_auction.row_generation.solve()

# Save theta_hat on rank 0
if rank == 0:
    output_path = BASE_DIR / "theta_hat_real.npy"
    
    # Check if file exists and ask to overwrite
    if output_path.exists():
        print(f"\nWarning: {output_path} already exists")
        print("Overwriting with new estimation...")
    
    np.save(output_path, theta_hat)
    print(f"\n✓ Saved estimated parameters to: {output_path}")
    print(f"  Shape: {theta_hat.shape}")
    print(f"  Min: {theta_hat.min():.4f}, Max: {theta_hat.max():.4f}")
    print(f"  Mean: {theta_hat.mean():.4f}, Std: {theta_hat.std():.4f}")
    print(f"\nFirst 5 parameters: {theta_hat[:5]}")
    print(f"Last 5 parameters: {theta_hat[-5:]}")

