#!/bin/env python
"""Compute SE for non-FE parameters only (4 params instead of 497).

Non-FE indices: [0, 494, 495, 496]
- 0: modular agent feature
- 494-496: quadratic features

Usage:
    srun ./run-gurobi.bash python compute_se_non_fe.py --delta 4
    srun ./run-gurobi.bash python compute_se_non_fe.py --delta 2
"""

import argparse
import csv
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import time
from datetime import datetime
from pathlib import Path
from bundlechoice import BundleChoice
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parse arguments on all ranks (MPI-safe)
parser = argparse.ArgumentParser(description="Compute SE for non-FE parameters")
parser.add_argument("--delta", "-d", type=int, choices=[2, 4], required=True,
                    help="Distance parameter delta (must match theta_hat.csv)")
args = parser.parse_args()
DELTA = args.delta

if rank == 0:
    start_time = time.time()

BASE_DIR = os.path.dirname(__file__)
NUM_SIMULS = 200  # More simulations for stability
TIMELIMIT_SEC = 30
STEP_SIZE = 1e-4

# Non-FE parameter indices
NON_FE_INDICES = np.array([0, 494, 495, 496], dtype=np.int64)
PARAM_NAMES = ["bidder_elig_pop", "pop_distance", "travel_survey", "air_travel"]


def load_theta_from_csv(delta):
    """Load theta from theta_hat.csv for given delta (most recent row)."""
    csv_path = os.path.join(BASE_DIR, "estimation_results", "theta_hat.csv")
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find rows with matching delta
    matching = [r for r in rows if str(r.get("delta", "")) == str(delta)]
    if not matching:
        return None
    
    # Get the most recent row (last one)
    row = matching[-1]
    
    # Extract all theta values
    num_features = int(row.get("num_features", 497))
    theta = np.zeros(num_features)
    for i in range(num_features):
        key = f"theta_{i}"
        if key in row and row[key]:
            theta[i] = float(row[key])
    
    return theta


# Load theta_hat from CSV
if rank == 0:
    print("=" * 60)
    print(f"COMPUTING STANDARD ERRORS FOR δ = {DELTA}")
    print("=" * 60)
    
    # Try to load from CSV first
    theta_hat = load_theta_from_csv(DELTA)
    
    if theta_hat is not None:
        print(f"Loaded theta from theta_hat.csv (delta={DELTA})")
        print(f"  Shape: {theta_hat.shape}")
        print(f"  Non-FE values: {theta_hat[NON_FE_INDICES]}")
    else:
        # Fallback to theta.npy (for backward compatibility)
        theta_path = os.path.join(BASE_DIR, "estimation_results", "theta.npy")
        if os.path.exists(theta_path):
            theta_hat = np.load(theta_path)
            print(f"Warning: No theta found in CSV for delta={DELTA}")
            print(f"  Falling back to theta.npy (may not match delta!)")
        else:
            print(f"Error: No theta estimates found for delta={DELTA}")
            print(f"  Run estimation first: sbatch auction.sbatch {DELTA}")
            sys.exit(1)
    
    print(f"\nNon-FE parameters: {NON_FE_INDICES}")
else:
    theta_hat = None

# Load data from delta-specific directory
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data", f"delta{DELTA}")
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input data not found at {INPUT_DIR}")
        print(f"  Run: ./run-gurobi.bash python prepare_data.py --delta {DELTA}")
        sys.exit(1)
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))
    quadratic = np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy"))
    weights = np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    modular_agent = np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy"))
    capacity = np.load(os.path.join(INPUT_DIR, "capacity_i.npy"))
    
    num_agents = capacity.shape[0]
    num_items = weights.shape[0]
    num_features = modular_agent.shape[2] + num_items + quadratic.shape[2]
    
    input_data = {
        "item_data": {"modular": -np.eye(num_items), "quadratic": quadratic, "weights": weights},
        "agent_data": {"modular": modular_agent, "capacity": capacity},
        "errors": np.random.normal(0, 1, (num_agents, num_items)),
        "obs_bundle": obs_bundle
    }
    
    print(f"Problem: {num_agents} agents, {num_items} items, {num_features} features")
else:
    input_data = None
    num_agents = None
    num_items = None
    num_features = None

num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)

# Initialize BundleChoice
config = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": 1
    },
    "subproblem": {
        "name": "QuadKnapsack",
        "settings": {"TimeLimit": TIMELIMIT_SEC, "MIPGap_tol": 1e-2}
    },
    "standard_errors": {
        "num_simulations": NUM_SIMULS,
        "step_size": STEP_SIZE,
        "seed": 1995,
    }
}

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

if rank == 0:
    print(f"\nMPI: {comm.Get_size()} ranks")
    print(f"Simulations: {NUM_SIMULS}")
    print(f"Step size: {STEP_SIZE}")
    print(f"Computing SE for {len(NON_FE_INDICES)} non-FE parameters only")

# Compute SE for non-FE parameters only
se_result = bc.standard_errors.compute(
    theta_hat=theta_hat,
    num_simulations=NUM_SIMULS,
    step_size=STEP_SIZE,
    beta_indices=NON_FE_INDICES,
    seed=1995,
    optimize_for_subset=True,
)

if rank == 0 and se_result is not None:
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print(f"NON-FE STANDARD ERRORS (δ = {DELTA})")
    print("="*60)
    print(f"A matrix shape: {se_result.A_matrix.shape}")
    print(f"B matrix shape: {se_result.B_matrix.shape}")
    print(f"A condition number: {np.linalg.cond(se_result.A_matrix):.2e}")
    print(f"B condition number: {np.linalg.cond(se_result.B_matrix):.2e}")
    
    print("\nResults:")
    for i, idx in enumerate(NON_FE_INDICES):
        theta_val = theta_hat[idx]
        se_val = se_result.se[i]
        t_val = se_result.t_stats[i]
        print(f"  θ[{idx}] = {theta_val:.4f}, SE = {se_val:.4f}, t = {t_val:.2f}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "se_non_fe.npy"), se_result.se)
    np.save(os.path.join(OUTPUT_DIR, "A_non_fe.npy"), se_result.A_matrix)
    np.save(os.path.join(OUTPUT_DIR, "B_non_fe.npy"), se_result.B_matrix)
    
    # Save to CSV with metadata
    CSV_PATH = os.path.join(OUTPUT_DIR, "se_non_fe.csv")
    num_mpi = comm.Get_size()
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Prepare row data
    row_data = {
        "timestamp": timestamp,
        "delta": DELTA,
        "num_mpi": num_mpi,
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations_se": NUM_SIMULS,
        "step_size": STEP_SIZE,
        "total_time_sec": total_time,
        "A_cond_number": np.linalg.cond(se_result.A_matrix),
        "B_cond_number": np.linalg.cond(se_result.B_matrix),
    }
    
    # Add theta, SE, and t-stats for each non-FE parameter
    for i, (idx, name) in enumerate(zip(NON_FE_INDICES, PARAM_NAMES)):
        row_data[f"theta_{name}"] = theta_hat[idx]
        row_data[f"se_{name}"] = se_result.se[i]
        row_data[f"t_{name}"] = se_result.t_stats[i]
    
    # Write to CSV (append if exists)
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"\nSaved to: {OUTPUT_DIR}/")
    print(f"  - se_non_fe.npy")
    print(f"  - A_non_fe.npy")
    print(f"  - B_non_fe.npy")
    print(f"  - se_non_fe.csv (delta={DELTA})")
    
    print(f"\n" + "="*60)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)
