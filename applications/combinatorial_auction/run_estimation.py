#!/bin/env python
"""
Main estimation script for combinatorial auction v2.

Uses CSV-based input data with automatic feature naming from data files.

Usage:
    srun ./run-gurobi.bash python run_estimation.py --delta 4
    srun ./run-gurobi.bash python run_estimation.py --delta 2
    srun ./run-gurobi.bash python run_estimation.py --delta 4 --winners-only
"""

import argparse
import sys
import os
import json

# Add project root to Python path
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bundlechoice import BundleChoice
from bundlechoice.estimation import adaptive_gurobi_timeout
import numpy as np
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

USE_PREVIOUS_THETA = True
THETA_PATH = os.path.join(BASE_DIR, "estimation_results", "theta.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")


def get_input_dir(delta, winners_only, hq_distance=False):
    """Build input directory name from parameters."""
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(BASE_DIR, "input_data", suffix)


# =============================================================================
# Initialize BundleChoice
# =============================================================================
bc = BundleChoice()
bc.load_config(CONFIG_PATH)

# =============================================================================
# Load Data (auto-detects CSV/NPY, auto-names features from data sources)
# =============================================================================
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
    
    # Load data from directory - auto-detects CSV, extracts feature names
    input_data = bc.data.load_from_directory(INPUT_DIR, error_seed=1995)
    
    # Add item modular features (negative identity for FE)
    num_items = bc.config.dimensions.num_items
    input_data["item_data"]["modular"] = -np.eye(num_items)
    
    # Feature names are now auto-derived from CSV headers/metadata
    print(f"Loaded data from {INPUT_DIR}")
    print(f"  Agents: {bc.config.dimensions.num_agents}, Items: {num_items}")
    if bc.data.data_sources.get("agent_data", {}).get("modular"):
        print(f"  Modular features: {bc.data.data_sources['agent_data']['modular']}")
    if bc.data.data_sources.get("item_data", {}).get("quadratic"):
        print(f"  Quadratic features: {bc.data.data_sources['item_data']['quadratic']}")
else:
    input_data = None

# Broadcast dimensions
num_features = comm.bcast(bc.config.dimensions.num_features if rank == 0 else None, root=0)
num_items = comm.bcast(bc.config.dimensions.num_items if rank == 0 else None, root=0)
num_agents = comm.bcast(bc.config.dimensions.num_agents if rank == 0 else None, root=0)

# Update dimensions on non-root ranks
if rank != 0:
    bc.config.dimensions.num_features = num_features
    bc.config.dimensions.num_items = num_items
    bc.config.dimensions.num_agents = num_agents

# Load and scatter data, build features (auto-names features from data sources)
bc.data.load_and_scatter(input_data)
bc.oracles.build_from_data()
bc.subproblems.load()

# Broadcast feature names after build_from_data sets them
feature_names = comm.bcast(bc.config.dimensions.feature_names if rank == 0 else None, root=0)
if rank != 0:
    bc.config.dimensions.feature_names = feature_names

if rank == 0:
    print(f"Features: {num_features} total")
    if feature_names:
        structural = [n for n in feature_names if not n.startswith("FE_")]
        print(f"  Structural: {structural}")

# =============================================================================
# Custom Bounds for delta=2
# =============================================================================
if DELTA == 2 and feature_names:
    # Build bounds arrays by feature name
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
        print(f"Custom bounds for delta=2 applied")

# =============================================================================
# Adaptive Gurobi Timeout (fast early iterations, precise later)
# =============================================================================
adaptive_callback = adaptive_gurobi_timeout(
    initial_timeout=1.0,
    final_timeout=30.0,
    transition_iterations=15,
    strategy="linear",
    log=True
)
bc.config.row_generation.subproblem_callback = adaptive_callback

# =============================================================================
# Load Previous Theta (warm start)
# =============================================================================
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

# =============================================================================
# Run Estimation
# =============================================================================
result = bc.row_generation.solve(theta_init=theta_init)

# =============================================================================
# Save Results
# =============================================================================
if rank == 0:
    print(f"\n{result.summary()}")
    
    result.save_npy(os.path.join(OUTPUT_DIR, "theta.npy"))
    
    result.export_csv(
        os.path.join(OUTPUT_DIR, "theta_hat.csv"),
        metadata={
            "delta": DELTA,
            "winners_only": WINNERS_ONLY,
            "hq_distance": HQ_DISTANCE,
            "num_mpi": comm.Get_size(),
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": bc.config.dimensions.num_simulations,
        },
        feature_names=feature_names,
        append=True,
    )
    
    # Save metadata JSON
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
