#!/bin/env python
"""Compute SE for structural (non-FE) parameters using Bayesian Bootstrap.

Uses model warm-start for significant speedup (~20x vs baseline).

Usage:
    srun ./run-gurobi.bash python compute_se_bayesian_boot.py --delta 4
    srun ./run-gurobi.bash python compute_se_bayesian_boot.py --delta 2 --num-bootstrap 100
"""

import argparse
import csv
import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from bundlechoice import BundleChoice
from bundlechoice.estimation import adaptive_gurobi_timeout
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parse arguments
parser = argparse.ArgumentParser(description="Compute SE using Bayesian Bootstrap")
parser.add_argument("--delta", "-d", type=int, choices=[2, 4], required=True)
parser.add_argument("--winners-only", "-w", action="store_true")
parser.add_argument("--hq-distance", action="store_true")
parser.add_argument("--num-bootstrap", "-n", type=int, default=200,
                    help="Number of bootstrap samples (default: 200)")
parser.add_argument("--warmstart", choices=["none", "theta", "model", "model_strip"],
                    default="model_strip", help="Warm-start strategy (default: model_strip)")
parser.add_argument("--seed", type=int, default=1995, help="Random seed")
args = parser.parse_args()

DELTA = args.delta
WINNERS_ONLY = args.winners_only
HQ_DISTANCE = args.hq_distance
NUM_BOOTSTRAP = args.num_bootstrap
WARMSTART = args.warmstart
SEED = args.seed

if rank == 0:
    start_time = time.time()

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
TIMELIMIT_SEC = 10


def get_input_dir(delta, winners_only, hq_distance=False):
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(BASE_DIR, "input_data", suffix)


def load_theta_from_csv(delta, winners_only=False, hq_distance=False):
    """Load theta from theta_hat.csv for given parameters."""
    csv_path = os.path.join(BASE_DIR, "estimation_results", "theta_hat.csv")
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    
    def str_to_bool(s):
        """Convert string to bool, treating empty string as False."""
        if not s or s.lower() in ('', 'false', '0', 'no'):
            return False
        return True
    
    matching = [r for r in rows 
                if str(r.get("delta", "")) == str(delta)
                and str_to_bool(r.get("winners_only", "")) == winners_only
                and str_to_bool(r.get("hq_distance", "")) == hq_distance]
    
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


# =============================================================================
# Load Theta
# =============================================================================
if rank == 0:
    print("=" * 70)
    print("BAYESIAN BOOTSTRAP STANDARD ERRORS")
    print("=" * 70)
    print(f"  δ = {DELTA}, winners_only = {WINNERS_ONLY}, hq_distance = {HQ_DISTANCE}")
    print(f"  Bootstrap samples: {NUM_BOOTSTRAP}, Warm-start: {WARMSTART}")
    print(f"  Seed: {SEED}")
    print("=" * 70)
    
    theta_hat = load_theta_from_csv(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if theta_hat is None:
        print(f"Error: No theta estimates found in estimation_results/theta_hat.csv")
        sys.exit(1)
    print(f"Loaded theta: shape={theta_hat.shape}")
else:
    theta_hat = None

theta_hat = comm.bcast(theta_hat, root=0)

# =============================================================================
# Initialize BundleChoice and Load Data
# =============================================================================
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

bc = BundleChoice()
bc.load_config(CONFIG_PATH)

# Use 10 simulations for reduced MC variance in bootstrap
bc.load_config({"dimensions": {"num_simulations": 10}})

if rank == 0:
    INPUT_DIR = get_input_dir(DELTA, WINNERS_ONLY, HQ_DISTANCE)
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input data not found at {INPUT_DIR}")
        print(f"  Run: ./run-gurobi.bash python prepare_data.py --delta {DELTA}")
        sys.exit(1)
    
    # Load data (auto-detects CSV, extracts feature names)
    input_data = bc.data.load_from_directory(INPUT_DIR, error_seed=SEED)
    
    # Add item modular features
    num_items = bc.config.dimensions.num_items
    input_data["item_data"]["modular"] = -np.eye(num_items)
else:
    input_data = None

# Broadcast dimensions
num_features = comm.bcast(bc.config.dimensions.num_features if rank == 0 else None, root=0)
num_items = comm.bcast(bc.config.dimensions.num_items if rank == 0 else None, root=0)
num_agents = comm.bcast(bc.config.dimensions.num_agents if rank == 0 else None, root=0)
num_simulations = comm.bcast(bc.config.dimensions.num_simulations if rank == 0 else None, root=0)

if rank != 0:
    bc.config.dimensions.num_features = num_features
    bc.config.dimensions.num_items = num_items
    bc.config.dimensions.num_agents = num_agents
    bc.config.dimensions.num_simulations = num_simulations

# Update config for this run (must include num_simulations to avoid default=1 overwrite)
bc.load_config({
    "dimensions": {"num_simulations": num_simulations},
    "subproblem": {"name": "QuadKnapsack", "settings": {"TimeLimit": TIMELIMIT_SEC, "MIPGap_tol": 1e-2}},
    "row_generation": {"max_iters": 200, "tolerance_optimality": 0.01},
})

# Adaptive timeout: fast cuts early, precise cuts near convergence
adaptive_callback = adaptive_gurobi_timeout(
    initial_timeout=1.0,
    final_timeout=30.0,
    transition_iterations=15,
    strategy="linear",
    log=True
)
bc.config.row_generation.subproblem_callback = adaptive_callback

# Set lower bounds for travel_survey and air_travel parameters
theta_lbs = np.zeros(num_features)
theta_lbs[-1] = -150  # air_travel
theta_lbs[-2] = -150  # travel_survey
bc.config.row_generation.theta_lbs = theta_lbs

# Custom constraint: pop_distance >= travel_survey + air_travel
def add_custom_constraints(model, theta, u):
    model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")

bc.config.row_generation.master_init_callback = add_custom_constraints

bc.data.load_and_scatter(input_data)
bc.oracles.build_from_data()
bc.subproblems.load()

# Broadcast feature names
feature_names = comm.bcast(bc.config.dimensions.feature_names if rank == 0 else None, root=0)
if rank != 0:
    bc.config.dimensions.feature_names = feature_names

# Get structural indices (non-FE parameters only)
if feature_names:
    structural_indices = np.array([i for i, name in enumerate(feature_names) 
                                   if not name.startswith("FE_")], dtype=np.int64)
    structural_names = [feature_names[i] for i in structural_indices]
else:
    structural_indices = np.array(bc.config.dimensions.get_structural_indices(), dtype=np.int64)
    structural_names = [bc.config.dimensions.get_feature_name(i) for i in structural_indices]

if rank == 0:
    print(f"\nProblem: {num_agents} agents, {num_items} items, {num_features} features")
    print(f"Structural parameters ({len(structural_indices)}): {structural_names}")
    print(f"MPI ranks: {comm.Get_size()}")
    print()

# =============================================================================
# Initial solve (needed for model warm-start)
# =============================================================================
if rank == 0:
    print("Running initial solve to setup model for warm-start...")

result = bc.row_generation.solve(theta_init=theta_hat)

if rank == 0:
    print(f"Initial solve complete: {result.num_iterations} iterations")
    print()

# =============================================================================
# Compute Standard Errors via Bayesian Bootstrap
# =============================================================================
if rank == 0:
    print(f"Starting Bayesian bootstrap ({NUM_BOOTSTRAP} samples)...")
    boot_start = time.time()

se_result = bc.standard_errors.compute_bayesian_bootstrap(
    theta_hat=theta_hat,
    row_generation=bc.row_generation,
    num_bootstrap=NUM_BOOTSTRAP,
    beta_indices=structural_indices,
    seed=SEED,
    warmstart=WARMSTART,
)

if rank == 0:
    boot_time = time.time() - boot_start
    print(f"Bayesian bootstrap complete: {boot_time:.1f}s")

# =============================================================================
# Save Results
# =============================================================================
if rank == 0 and se_result is not None:
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"BAYESIAN BOOTSTRAP RESULTS (δ = {DELTA})")
    print("=" * 70)
    
    print("\nStructural Parameters:")
    for i, name in enumerate(structural_names):
        idx = structural_indices[i]
        print(f"  {name}: θ={theta_hat[idx]:.4f}, SE={se_result.se[i]:.4f}, t={se_result.t_stats[i]:.2f}")
    
    # Save arrays
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "se_bayesian_boot.npy"), se_result.se)
    
    # Export to CSV
    CSV_PATH = os.path.join(OUTPUT_DIR, "se_bayesian_boot.csv")
    row_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "delta": DELTA,
        "winners_only": WINNERS_ONLY,
        "hq_distance": HQ_DISTANCE,
        "num_mpi": comm.Get_size(),
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_bootstrap": NUM_BOOTSTRAP,
        "warmstart": WARMSTART,
        "seed": SEED,
        "total_time_sec": total_time,
        "boot_time_sec": boot_time,
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
    
    print(f"\nResults saved to {OUTPUT_DIR}/ ({total_time:.1f}s total, {boot_time:.1f}s bootstrap)")
