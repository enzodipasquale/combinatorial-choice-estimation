#!/bin/env python
"""
Sandwich SE estimation.

Requires pre-computed theta from point_estimate.
All parameters are read from config.yaml (cluster) or config_local.yaml (local).

Usage:
    srun ./run-gurobi.bash python compute_se_sandwich.py
"""

import sys
import os
import csv
import time
import yaml
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from bundlechoice import BundleChoice
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load config
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters
app_cfg = config.get("application", {})
DELTA = app_cfg.get("delta", 4)
WINNERS_ONLY = app_cfg.get("winners_only", False)
HQ_DISTANCE = app_cfg.get("hq_distance", False)

sandwich_cfg = config.get("sandwich", {})
NUM_SIMULS = sandwich_cfg.get("num_simulations", 200)
STEP_SIZE = sandwich_cfg.get("step_size", 1e-4)
SEED = sandwich_cfg.get("seed", 1995)

OUTPUT_DIR = os.path.join(APP_DIR, "estimation_results")

if rank == 0:
    start_time = time.time()
    print("=" * 60)
    print("SANDWICH STANDARD ERRORS")
    print(f"  delta={DELTA}, winners_only={WINNERS_ONLY}, hq_distance={HQ_DISTANCE}")
    print(f"  Simulations: {NUM_SIMULS}, Step size: {STEP_SIZE}")
    print("=" * 60)


def get_input_dir(delta, winners_only, hq_distance=False):
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(APP_DIR, "data", "114402-V1", "input_data", suffix)


def load_theta_from_csv(delta, winners_only=False, hq_distance=False):
    """Load theta from theta_hat.csv for given parameters."""
    csv_path = os.path.join(APP_DIR, "estimation_results", "theta_hat.csv")
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    
    matching = [r for r in rows 
                if str(r.get("delta", "")) == str(delta)
                and str(r.get("winners_only", "False")).lower() == str(winners_only).lower()
                and str(r.get("hq_distance", "False")).lower() == str(hq_distance).lower()]
    
    if not matching:
        return None
    
    row = matching[-1]
    num_features = int(row.get("num_features", 497))
    theta = np.zeros(num_features)
    for i in range(num_features):
        key = f"theta_{i}"
        if key in row and row[key]:
            theta[i] = float(row[key])
    return theta


# Load theta
if rank == 0:
    theta_hat = load_theta_from_csv(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if theta_hat is None:
        print(f"Error: No theta estimates found for delta={DELTA}")
        print(f"  Run point_estimate first")
        sys.exit(1)
    print(f"Loaded theta: shape={theta_hat.shape}")
else:
    theta_hat = None

theta_hat = comm.bcast(theta_hat, root=0)

# Initialize BundleChoice
bc = BundleChoice()
bc_config = {k: v for k, v in config.items() 
             if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]}
bc.load_config(bc_config)

if rank == 0:
    INPUT_DIR = get_input_dir(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input data not found at {INPUT_DIR}")
        sys.exit(1)
    
    input_data = bc.data.load_from_directory(INPUT_DIR, error_seed=SEED)
    num_items = bc.config.dimensions.num_items
    input_data["item_data"]["modular"] = -np.eye(num_items)
else:
    input_data = None

# Broadcast
num_features = comm.bcast(bc.config.dimensions.num_features if rank == 0 else None, root=0)
num_items = comm.bcast(bc.config.dimensions.num_items if rank == 0 else None, root=0)
num_agents = comm.bcast(bc.config.dimensions.num_agents if rank == 0 else None, root=0)

if rank != 0:
    bc.config.dimensions.num_features = num_features
    bc.config.dimensions.num_items = num_items
    bc.config.dimensions.num_agents = num_agents

bc.data.load_and_scatter(input_data)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

# Broadcast feature names
feature_names = comm.bcast(bc.config.dimensions.feature_names if rank == 0 else None, root=0)
if rank != 0:
    bc.config.dimensions.feature_names = feature_names

# Get structural indices
structural_indices = np.array(bc.config.dimensions.get_structural_indices(), dtype=np.int64)
structural_names = [bc.config.dimensions.get_feature_name(i) for i in structural_indices]

if rank == 0:
    print(f"\nProblem: {num_agents} agents, {num_items} items")
    print(f"Structural parameters: {structural_names}")
    print(f"MPI: {comm.Get_size()} ranks")

# Compute SE
se_result = bc.standard_errors.compute(
    theta_hat=theta_hat,
    num_simulations=NUM_SIMULS,
    step_size=STEP_SIZE,
    beta_indices=structural_indices,
    seed=SEED,
    optimize_for_subset=True,
)

# Save results
if rank == 0 and se_result is not None:
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"RESULTS (delta={DELTA})")
    print("=" * 60)
    
    for i, name in enumerate(structural_names):
        idx = structural_indices[i]
        print(f"  {name}: theta={theta_hat[idx]:.4f}, SE={se_result.se[i]:.4f}, t={se_result.t_stats[i]:.2f}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save A and B matrices (not in CSV)
    np.save(os.path.join(OUTPUT_DIR, "A_sandwich.npy"), se_result.A_matrix)
    np.save(os.path.join(OUTPUT_DIR, "B_sandwich.npy"), se_result.B_matrix)
    
    CSV_PATH = os.path.join(OUTPUT_DIR, "se_sandwich.csv")
    row_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "delta": DELTA, "winners_only": WINNERS_ONLY, "hq_distance": HQ_DISTANCE,
        "num_mpi": comm.Get_size(), "num_agents": num_agents,
        "num_items": num_items, "num_features": num_features,
        "num_simulations_se": NUM_SIMULS, "step_size": STEP_SIZE,
        "total_time_sec": total_time,
        "A_cond_number": np.linalg.cond(se_result.A_matrix),
        "B_cond_number": np.linalg.cond(se_result.B_matrix),
    }
    for i, name in enumerate(structural_names):
        idx = structural_indices[i]
        row_data[f"theta_{name}"] = theta_hat[idx]
        row_data[f"se_{name}"] = se_result.se[i]
        row_data[f"t_{name}"] = se_result.t_stats[i]
    
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"\nSaved to {OUTPUT_DIR}/ ({total_time:.1f}s)")
