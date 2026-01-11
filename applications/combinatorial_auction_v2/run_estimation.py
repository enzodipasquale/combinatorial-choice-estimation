#!/bin/env python
"""
Main estimation script for combinatorial auction v2.

Usage:
    srun ./run-gurobi.bash python run_estimation.py --delta 4
    srun ./run-gurobi.bash python run_estimation.py --delta 2
    srun ./run-gurobi.bash python run_estimation.py --delta 4 --winners-only
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import csv
import json

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

# Parse arguments on all ranks (MPI-safe)
parser = argparse.ArgumentParser(description="Run estimation for combinatorial auction")
parser.add_argument("--delta", "-d", type=int, choices=[2, 4], required=True,
                    help="Distance parameter delta (must match prepare_data.py)")
parser.add_argument("--winners-only", "-w", action="store_true",
                    help="Use winners-only sample (must match prepare_data.py)")
parser.add_argument("--hq-distance", action="store_true",
                    help="Use HQ-to-item distance features (must match prepare_data.py)")
args = parser.parse_args()
DELTA = args.delta
WINNERS_ONLY = args.winners_only
HQ_DISTANCE = args.hq_distance

IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

# Option to use previous theta as initial value
USE_PREVIOUS_THETA = True
THETA_PATH = os.path.join(BASE_DIR, "estimation_results", "theta.npy")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Build input directory name from parameters
def get_input_dir(delta, winners_only, hq_distance=False):
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(BASE_DIR, "input_data", suffix)

# Load data on rank 0 from parameter-specific directory
if rank == 0:
    INPUT_DIR = get_input_dir(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if not os.path.exists(INPUT_DIR):
        cmd = f"./run-gurobi.bash python prepare_data.py --delta {DELTA}"
        if WINNERS_ONLY:
            cmd += " --winners-only"
        if HQ_DISTANCE:
            cmd += " --hq-distance"
        print(f"Error: Input data not found at {INPUT_DIR}")
        print(f"  Run: {cmd}")
        sys.exit(1)
    
    # Load metadata to get actual num_agents (varies with winners_only)
    with open(os.path.join(INPUT_DIR, "metadata.json"), "r") as f:
        input_metadata = json.load(f)
    
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    num_agents = input_metadata["num_agents"]  # From input data, not config
    num_items = config["dimensions"]["num_items"]
    num_simulations = config["dimensions"]["num_simulations"]
    # Compute num_features from metadata: modular + FE + quadratic
    num_modular = input_metadata.get("num_modular_features", 1)
    num_quadratic = input_metadata.get("num_quadratic_features", 3)
    num_features = num_modular + num_items + num_quadratic
    print(f"Features: {num_modular} modular + {num_items} FE + {num_quadratic} quadratic = {num_features}")

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

# Override config dimensions with actual data dimensions (varies with winners_only, hq_distance)
combinatorial_auction.config.dimensions.num_agents = num_agents
combinatorial_auction.config.dimensions.num_items = num_items
combinatorial_auction.config.dimensions.num_features = num_features
combinatorial_auction.config.dimensions.num_simulations = num_simulations

combinatorial_auction.data.load_and_scatter(input_data)
combinatorial_auction.features.build_from_data()
combinatorial_auction.subproblems.load()

# Custom bounds for delta=2 (per Fox & Bajari specification)
if DELTA == 2:
    theta_lbs = np.zeros(num_features)
    theta_ubs = np.full(num_features, 1000.0)
    # theta[0] >= 75 (modular parameter)
    theta_lbs[0] = 75
    # theta[-3] between 400 and 650 (pop/distance)
    theta_lbs[-3] = 400
    theta_ubs[-3] = 650
    # theta[-2] >= -120 (travel survey)
    theta_lbs[-2] = -120
    # theta[-1] >= -75 (air travel)
    theta_lbs[-1] = -75
    combinatorial_auction.config.row_generation.theta_lbs = theta_lbs
    combinatorial_auction.config.row_generation.theta_ubs = theta_ubs
    if rank == 0:
        print(f"Custom bounds for delta=2: theta_lbs[-3:]={theta_lbs[-3:]}, theta_ubs[-3:]={theta_ubs[-3:]}")

# Load previous theta if requested (only if dimensions match)
theta_init = None
if USE_PREVIOUS_THETA and os.path.exists(THETA_PATH):
    if rank == 0:
        prev_theta = np.load(THETA_PATH)
        if prev_theta.shape[0] == num_features:
            theta_init = prev_theta
            print(f"Loading previous theta from {THETA_PATH}")
        else:
            print(f"Skipping previous theta (dimension mismatch: {prev_theta.shape[0]} vs {num_features})")
    theta_init = comm.bcast(theta_init, root=0)

result = combinatorial_auction.row_generation.solve(theta_init=theta_init)

if rank == 0:
    print(f"\n{result.summary()}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Parameters from command-line
    delta = DELTA
    winners_only = WINNERS_ONLY
    
    # Save theta_hat as numpy array
    np.save(os.path.join(OUTPUT_DIR, "theta.npy"), result.theta_hat)
    
    # Save theta metadata (for SE computation)
    theta_metadata = {
        "delta": delta,
        "winners_only": winners_only,
        "hq_distance": HQ_DISTANCE,
        "num_features": num_features,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "converged": result.converged,
        "num_iterations": result.num_iterations,
    }
    with open(os.path.join(OUTPUT_DIR, "theta_metadata.json"), "w") as f:
        json.dump(theta_metadata, f, indent=2)
    
    # Save to CSV with metadata
    CSV_PATH = os.path.join(OUTPUT_DIR, "theta_hat.csv")
    num_mpi = comm.Get_size()
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Prepare row data
    row_data = {
        "timestamp": timestamp,
        "delta": delta,
        "winners_only": winners_only,
        "hq_distance": HQ_DISTANCE,
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
    
    # Write to CSV - handle case where new row has more columns than existing file
    if os.path.exists(CSV_PATH):
        # Read existing data to check columns
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames if reader.fieldnames else []
        
        # Check if we have new columns (filter out None values from existing)
        new_fieldnames = list(row_data.keys())
        all_fieldnames = [f for f in existing_fieldnames if f is not None]
        for col in new_fieldnames:
            if col not in all_fieldnames:
                all_fieldnames.append(col)
        
        # Always rewrite to ensure clean CSV (no None fields, consistent columns)
        print(f"  Writing CSV with {len(all_fieldnames)} columns")
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            for row in existing_rows:
                # Filter out None keys from existing rows
                clean_row = {k: v for k, v in row.items() if k is not None}
                writer.writerow(clean_row)
            writer.writerow(row_data)
    else:
        # Create new file
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            writer.writeheader()
            writer.writerow(row_data)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  - theta.npy")
    print(f"  - theta_hat.csv")
