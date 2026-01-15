#!/bin/env python
"""
Point estimate for combinatorial auction.

All parameters are read from config.yaml (cluster) or config_local.yaml (local).
No command-line arguments needed.

Usage:
    srun ./run-gurobi.bash python run_estimation.py
"""

import sys
import os
import json
import yaml

BASE_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bundlechoice import BundleChoice
from bundlechoice.estimation import adaptive_gurobi_timeout
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load config
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract application parameters
app_cfg = config.get("application", {})
DELTA = app_cfg.get("delta", 4)
WINNERS_ONLY = app_cfg.get("winners_only", False)
HQ_DISTANCE = app_cfg.get("hq_distance", False)
USE_PREVIOUS_THETA = app_cfg.get("use_previous_theta", True)

THETA_PATH = os.path.join(APP_DIR, "estimation_results", "theta.npy")
OUTPUT_DIR = os.path.join(APP_DIR, "estimation_results")


def get_input_dir(delta, winners_only, hq_distance=False):
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(APP_DIR, "data", "114402-V1", "input_data", suffix)


# Initialize BundleChoice with bundlechoice-specific config
bc = BundleChoice()
bc_config = {k: v for k, v in config.items() 
             if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]}
bc.load_config(bc_config)

# Load data
if rank == 0:
    INPUT_DIR = get_input_dir(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input data not found at {INPUT_DIR}")
        print(f"  Run: python ../data/prepare_data.py --delta {DELTA}")
        sys.exit(1)
    
    input_data = bc.data.load_from_directory(INPUT_DIR, error_seed=1995)
    
    num_items = bc.config.dimensions.num_items
    input_data["item_data"]["modular"] = -np.eye(num_items)
    
    print(f"Config: delta={DELTA}, winners_only={WINNERS_ONLY}, hq_distance={HQ_DISTANCE}")
    print(f"Loaded data from {INPUT_DIR}")
    print(f"  Agents: {bc.config.dimensions.num_obs}, Items: {num_items}")
else:
    input_data = None

# Broadcast dimensions
num_features = comm.bcast(bc.config.dimensions.num_features if rank == 0 else None, root=0)
num_items = comm.bcast(bc.config.dimensions.num_items if rank == 0 else None, root=0)
num_obs = comm.bcast(bc.config.dimensions.num_obs if rank == 0 else None, root=0)

if rank != 0:
    bc.config.dimensions.num_features = num_features
    bc.config.dimensions.num_items = num_items
    bc.config.dimensions.num_obs = num_obs

bc.data.load_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.subproblems.load()

feature_names = comm.bcast(bc.config.dimensions.feature_names if rank == 0 else None, root=0)
if rank != 0:
    bc.config.dimensions.feature_names = feature_names

if rank == 0:
    print(f"Features: {num_features} total")
    if feature_names:
        structural = [n for n in feature_names if not n.startswith("FE_")]
        print(f"  Structural: {structural}")

# Custom bounds for delta=2
if DELTA == 2 and feature_names:
    theta_lbs = np.zeros(num_features)
    theta_ubs = np.full(num_features, 1000.0)
    
    for i, name in enumerate(feature_names):
        if name == "bidder_elig_pop":
            theta_lbs[i] = 75
        elif name == "pop_distance":
            theta_lbs[i], theta_ubs[i] = 400, 650
        elif name == "travel_survey":
            theta_lbs[i] = -120
        elif name == "air_travel":
            theta_lbs[i] = -75
        elif name.startswith("FE_"):
            theta_lbs[i], theta_ubs[i] = 0, 1000
    
    bc.config.row_generation.theta_lbs = theta_lbs
    bc.config.row_generation.theta_ubs = theta_ubs
    if rank == 0:
        print("Custom bounds for delta=2 applied")

# Adaptive timeout
adaptive_cfg = config.get("adaptive_timeout", {})
if adaptive_cfg:
    adaptive_callback = adaptive_gurobi_timeout(
        initial_timeout=adaptive_cfg.get("initial", 1.0),
        final_timeout=adaptive_cfg.get("final", 30.0),
        transition_iterations=adaptive_cfg.get("transition_iterations", 15),
        strategy=adaptive_cfg.get("strategy", "linear"),
        log=True
    )
    bc.config.row_generation.subproblem_callback = adaptive_callback

# Warm start from previous theta
theta_init = None
if USE_PREVIOUS_THETA and os.path.exists(THETA_PATH):
    if rank == 0:
        prev_theta = np.load(THETA_PATH)
        if prev_theta.shape[0] == num_features:
            theta_init = prev_theta
            print(f"Loading previous theta from {THETA_PATH}")
        else:
            print(f"Skipping previous theta (dimension mismatch)")
    theta_init = comm.bcast(theta_init, root=0)

# Run estimation
result = bc.row_generation.solve(theta_init=theta_init)

# Save results
if rank == 0:
    print(f"\n{result.summary()}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.save_npy(os.path.join(OUTPUT_DIR, "theta.npy"))
    
    result.export_csv(
        os.path.join(OUTPUT_DIR, "theta_hat.csv"),
        metadata={
            "delta": DELTA,
            "winners_only": WINNERS_ONLY,
            "hq_distance": HQ_DISTANCE,
            "num_mpi": comm.Get_size(),
            "num_obs": num_obs,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": bc.config.dimensions.num_simulations,
        },
        feature_names=feature_names,
        append=True,
    )
    
    from datetime import datetime
    theta_metadata = {
        "delta": DELTA,
        "winners_only": WINNERS_ONLY,
        "hq_distance": HQ_DISTANCE,
        "num_features": num_features,
        "feature_names": feature_names,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "converged": result.converged,
        "num_iterations": result.num_iterations,
    }
    with open(os.path.join(OUTPUT_DIR, "theta_metadata.json"), "w") as f:
        json.dump(theta_metadata, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
