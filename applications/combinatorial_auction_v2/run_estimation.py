#!/bin/env python
"""
Main estimation script for combinatorial auction v2.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import csv

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

# Option to use previous theta as initial value
USE_PREVIOUS_THETA = True
THETA_PATH = os.path.join(BASE_DIR, "estimation_results", "theta.npy")

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

# Load previous theta if requested
theta_init = None
if USE_PREVIOUS_THETA and os.path.exists(THETA_PATH):
    if rank == 0:
        theta_init = np.load(THETA_PATH)
        print(f"Loading previous theta from {THETA_PATH}")
    theta_init = comm.bcast(theta_init, root=0)

result = combinatorial_auction.row_generation.solve(theta_init=theta_init)

if rank == 0:
    print(f"\n{result.summary()}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save theta_hat as numpy array
    np.save(os.path.join(OUTPUT_DIR, "theta.npy"), result.theta_hat)
    
    # Save to CSV with metadata
    CSV_PATH = os.path.join(OUTPUT_DIR, "theta_hat.csv")
    num_mpi = comm.Get_size()
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Prepare row data
    row_data = {
        "timestamp": timestamp,
        "num_mpi": num_mpi,
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "converged": result.converged,
        "num_iterations": result.num_iterations,
        "final_objective": result.final_objective if result.final_objective is not None else "",
    }
    
    # Add timing statistics if available
    if result.timing:
        timing = result.timing
        row_data.update({
            "total_time": timing.get("total_time", ""),
            "time_per_iter": timing.get("time_per_iter", ""),
            "pricing_time": timing.get("pricing_time", ""),
            "pricing_pct": timing.get("pricing_pct", ""),
            "master_time": timing.get("master_time", ""),
            "master_pct": timing.get("master_pct", ""),
            "other_time": timing.get("other_time", ""),
            "other_pct": timing.get("other_pct", ""),
        })
    else:
        row_data.update({
            "total_time": "",
            "time_per_iter": "",
            "pricing_time": "",
            "pricing_pct": "",
            "master_time": "",
            "master_pct": "",
            "other_time": "",
            "other_pct": "",
        })
    
    # Add theta values as separate columns
    for i, theta_val in enumerate(result.theta_hat):
        row_data[f"theta_{i}"] = theta_val
    
    # Write to CSV (append if file exists, create with headers if not)
    file_exists = os.path.exists(CSV_PATH)
    
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  - theta.npy")
    print(f"  - theta_hat.csv")
