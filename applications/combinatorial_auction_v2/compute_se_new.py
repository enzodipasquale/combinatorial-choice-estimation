#!/bin/env python
"""Compute sandwich SE using the new bc.standard_errors API.

HPC version - computes full SE for all 497 parameters.
Expected runtime: 3-4 hours with 200 MPI ranks.
"""

import sys
import os

# Add bundlechoice to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from bundlechoice import BundleChoice
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
NUM_SIMULS_SE = 10   # Production setting
STEP_SIZE = 1e-4     # Smaller step for accuracy
TIMELIMIT_SEC = 1.0  # Production timeout

# Load theta_hat
if rank == 0:
    theta_hat = np.load(os.path.join(BASE_DIR, "estimation_results", "theta.npy"))
    print(f"Loaded theta_hat: shape={theta_hat.shape}")
else:
    theta_hat = None

# Load data
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
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
        "num_simuls": 1
    },
    "subproblem": {
        "name": "QuadKnapsack",
        "settings": {"TimeLimit": TIMELIMIT_SEC, "MIPGap_tol": 1e-2}
    },
    "standard_errors": {
        "num_simulations": NUM_SIMULS_SE,
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
    print(f"Simulations: {NUM_SIMULS_SE}")
    print(f"Step size: {STEP_SIZE}")
    print(f"Computing FULL SE for all {num_features} parameters")

# Compute SE for ALL parameters (no beta_indices = full computation)
se_result = bc.standard_errors.compute(
    theta_hat=theta_hat,
    optimize_for_subset=False,  # Full matrices
)

if rank == 0:
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    
    # Save full results
    np.save(os.path.join(OUTPUT_DIR, "sandwich_se_all.npy"), se_result.se_all)
    np.save(os.path.join(OUTPUT_DIR, "sandwich_variance.npy"), se_result.variance)
    np.save(os.path.join(OUTPUT_DIR, "sandwich_A_matrix.npy"), se_result.A_matrix)
    np.save(os.path.join(OUTPUT_DIR, "sandwich_B_matrix.npy"), se_result.B_matrix)
    
    # Create CSV with results
    df = pd.DataFrame({
        "feature_index": np.arange(num_features),
        "theta": theta_hat,
        "se": se_result.se_all,
        "t_stat": np.where(se_result.se_all > 0, theta_hat / se_result.se_all, 0.0),
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "sandwich_se_full.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(f"  SE array: {OUTPUT_DIR}/sandwich_se_all.npy")
    print(f"  Variance matrix: {OUTPUT_DIR}/sandwich_variance.npy")
    print(f"  A matrix: {OUTPUT_DIR}/sandwich_A_matrix.npy")
    print(f"  B matrix: {OUTPUT_DIR}/sandwich_B_matrix.npy")
    print(f"  CSV: {OUTPUT_DIR}/sandwich_se_full.csv")
    
    # Print summary for non-FE parameters
    info = bc.data_manager.get_data_info()
    num_modular_agent = info["num_modular_agent"]
    num_modular_item = info["num_modular_item"]
    beta_indices = np.concatenate([
        np.arange(num_modular_agent),
        np.arange(num_modular_agent + num_modular_item, num_features)
    ])
    
    print("\n" + "=" * 60)
    print("NON-FE PARAMETERS SUMMARY")
    print("=" * 60)
    for idx in beta_indices:
        print(f"  θ[{idx}] = {theta_hat[idx]:.4f}, SE = {se_result.se_all[idx]:.4f}, t = {theta_hat[idx]/se_result.se_all[idx]:.2f}")
    
    print("\n✓ DONE")
