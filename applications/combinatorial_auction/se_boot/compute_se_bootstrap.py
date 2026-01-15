#!/bin/env python
"""
Bootstrap SE estimation - fully independent pipeline.

Runs estimation first (to get point estimate), then computes bootstrap SE.
All parameters are read from config.yaml (cluster) or config_local.yaml (local).

Usage:
    srun ./run-gurobi.bash python compute_se_bootstrap.py
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
from bundlechoice.estimation import adaptive_gurobi_timeout
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

boot_cfg = config.get("bootstrap", {})
NUM_BOOTSTRAP = boot_cfg.get("num_samples", 200)
WARMSTART = boot_cfg.get("warmstart", "model_strip")
SEED = boot_cfg.get("seed", 1995)
INITIAL_ESTIMATION = boot_cfg.get("initial_estimation", True)

OUTPUT_DIR = os.path.join(APP_DIR, "estimation_results")

if rank == 0:
    start_time = time.time()
    print("=" * 70)
    print("BOOTSTRAP SE ESTIMATION (Independent Pipeline)")
    print("=" * 70)
    print(f"  Config: {CONFIG_PATH}")
    print(f"  delta={DELTA}, winners_only={WINNERS_ONLY}, hq_distance={HQ_DISTANCE}")
    print(f"  Bootstrap: {NUM_BOOTSTRAP} samples, warmstart={WARMSTART}, seed={SEED}")
    print(f"  Initial estimation: {INITIAL_ESTIMATION}")
    print("=" * 70)


def get_input_dir(delta, winners_only, hq_distance=False):
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return os.path.join(APP_DIR, "data", "114402-V1", "input_data", suffix)


# Initialize BundleChoice (load_config auto-broadcasts dimensions to all ranks)
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
    
    input_data = bc.data.load_from_directory(INPUT_DIR, error_seed=SEED)
    input_data["item_data"]["modular"] = -np.eye(bc.num_items)
    
    print(f"\nLoaded data from {INPUT_DIR}")
else:
    input_data = None

# Apply theta bounds from config
theta_bounds = config.get("theta_bounds", {})
if theta_bounds:
    theta_lbs = np.zeros(bc.num_features)
    if "air_travel_lb" in theta_bounds:
        theta_lbs[-1] = theta_bounds["air_travel_lb"]
    if "travel_survey_lb" in theta_bounds:
        theta_lbs[-2] = theta_bounds["travel_survey_lb"]
    bc.config.row_generation.theta_lbs = theta_lbs

# Custom constraints from config
constraints_cfg = config.get("constraints", {})
if constraints_cfg.get("pop_dominates_travel", False):
    def add_custom_constraints(model, theta, u):
        model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
    bc.config.row_generation.master_init_callback = add_custom_constraints

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

bc.data.load_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.subproblems.load()

# Get structural indices (non-FE parameters) - uses DimensionsConfig method
structural_indices = np.array(bc.config.dimensions.get_index_by_name(), dtype=np.int64)
structural_names = [bc.config.dimensions.get_feature_name(i) for i in structural_indices]

if rank == 0:
    print(f"Problem: {bc.num_obs} agents, {bc.num_items} items, {bc.num_features} features")
    print(f"Structural parameters ({len(structural_indices)}): {structural_names}")
    print(f"MPI ranks: {comm.Get_size()}")

# =============================================================================
# Step 1: Run estimation to get point estimate (optional)
# =============================================================================
if INITIAL_ESTIMATION:
    if rank == 0:
        print("\n" + "-" * 70)
        print("STEP 1: Point Estimation")
        print("-" * 70)

    result = bc.row_generation.solve()
    theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)

    if rank == 0:
        print(f"Estimation complete: {result.num_iterations} iterations")
        print(f"Theta (structural): {theta_hat[structural_indices]}")
else:
    result = None
    theta_hat = None
    if rank == 0:
        print("\n[Skipping initial estimation - bootstrap will use mean as point estimate]")

# =============================================================================
# Step 2: Compute Bootstrap SE
# =============================================================================
if rank == 0:
    print("\n" + "-" * 70)
    print(f"STEP 2: Bootstrap SE ({NUM_BOOTSTRAP} samples)")
    print("-" * 70)
    boot_start = time.time()

se_result = bc.standard_errors.compute_bayesian_bootstrap(
    row_generation=bc.row_generation,
    num_bootstrap=NUM_BOOTSTRAP,
    beta_indices=structural_indices,
    seed=SEED,
    warmstart=WARMSTART,
    theta_hat=theta_hat,
    initial_estimation=INITIAL_ESTIMATION,
)

if rank == 0:
    boot_time = time.time() - boot_start
    print(f"Bootstrap complete: {boot_time:.1f}s")

# =============================================================================
# Save Results
# =============================================================================
if rank == 0 and se_result is not None:
    total_time = time.time() - start_time
    
    # Use se_result.theta_hat for point estimate (works for both initial_estimation modes)
    theta_point = se_result.theta_hat
    
    print("\n" + "=" * 70)
    print(f"RESULTS (delta={DELTA})")
    print("=" * 70)
    
    print("\nStructural Parameters:")
    for i, name in enumerate(structural_names):
        idx = structural_indices[i]
        print(f"  {name}: theta={theta_point[idx]:.4f}, SE={se_result.se[i]:.4f}, t={se_result.t_stats[i]:.2f}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Export to CSV (primary output)
    CSV_PATH = os.path.join(OUTPUT_DIR, "se_bootstrap.csv")
    row_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "delta": DELTA,
        "winners_only": WINNERS_ONLY,
        "hq_distance": HQ_DISTANCE,
        "num_mpi": comm.Get_size(),
        "num_obs": bc.num_obs,
        "num_items": bc.num_items,
        "num_features": bc.num_features,
        "num_simulations": bc.num_simulations,
        "num_bootstrap": NUM_BOOTSTRAP,
        "warmstart": WARMSTART,
        "seed": SEED,
        "initial_estimation": INITIAL_ESTIMATION,
        "total_time_sec": total_time,
        "boot_time_sec": boot_time,
        "converged": result.converged if result else None,
        "num_iterations": result.num_iterations if result else None,
    }
    for i, name in enumerate(structural_names):
        idx = structural_indices[i]
        row_data[f"theta_{name}"] = theta_point[idx]
        row_data[f"se_{name}"] = se_result.se[i]
        row_data[f"t_{name}"] = se_result.t_stats[i]
    
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"Total time: {total_time:.1f}s (estimation + {boot_time:.1f}s bootstrap)")

# Handle bootstrap failure
if rank == 0 and se_result is None:
    print("\nERROR: Bootstrap failed - no successful samples")
    print("Check error messages above for details")

# Ensure clean MPI exit
comm.Barrier()
if rank == 0:
    print("\nDone.")
