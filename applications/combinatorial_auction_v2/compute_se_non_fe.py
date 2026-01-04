#!/bin/env python
"""Compute SE for non-FE parameters only (4 params instead of 497).

Non-FE indices: [0, 494, 495, 496]
- 0: modular agent feature
- 494-496: quadratic features

Run with: mpirun -n 12 python compute_se_non_fe.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import time
from bundlechoice import BundleChoice
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    start_time = time.time()

BASE_DIR = os.path.dirname(__file__)
NUM_SIMULS = 200  # More simulations for stability
TIMELIMIT_SEC = 1.0
STEP_SIZE = 1e-4

# Non-FE parameter indices
# Feature structure: [modular_agent(1), item_FE(493), quadratic(3)]
NON_FE_INDICES = np.array([0, 494, 495, 496], dtype=np.int64)

# Load theta_hat
if rank == 0:
    theta_hat = np.load(os.path.join(BASE_DIR, "estimation_results", "theta.npy"))
    print(f"Loaded theta_hat: shape={theta_hat.shape}")
    print(f"Non-FE parameters: {NON_FE_INDICES}")
    print(f"Non-FE theta values: {theta_hat[NON_FE_INDICES]}")
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
    print(f"Using optimize_for_subset=True for faster computation")

# Compute SE for non-FE parameters only
se_result = bc.standard_errors.compute(
    theta_hat=theta_hat,
    num_simulations=NUM_SIMULS,
    step_size=STEP_SIZE,
    beta_indices=NON_FE_INDICES,
    seed=1995,
    optimize_for_subset=True,  # Only compute 4x4 matrices
)

if rank == 0 and se_result is not None:
    print("\n" + "="*60)
    print("NON-FE STANDARD ERRORS (SUBSET COMPUTATION)")
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
    
    # Compare with previous HPC results
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS HPC RUN (10 sims, full 497x497)")
    print("="*60)
    prev_se = {
        0: 232.714023,
        494: 55.782308,
        495: 1101.776229,
        496: 822.364941,
    }
    print(f"{'Param':<8} {'Old SE':>12} {'New SE':>12} {'Ratio':>10}")
    print("-"*45)
    for i, idx in enumerate(NON_FE_INDICES):
        old = prev_se[idx]
        new = se_result.se[i]
        ratio = new / old if old > 0 else float('inf')
        print(f"θ[{idx}]    {old:>12.4f} {new:>12.4f} {ratio:>10.4f}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    np.save(os.path.join(OUTPUT_DIR, "se_non_fe.npy"), se_result.se)
    np.save(os.path.join(OUTPUT_DIR, "A_non_fe.npy"), se_result.A_matrix)
    np.save(os.path.join(OUTPUT_DIR, "B_non_fe.npy"), se_result.B_matrix)
    print(f"\nSaved to: {OUTPUT_DIR}/se_non_fe.npy")
    
    # Print total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n" + "="*60)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)


