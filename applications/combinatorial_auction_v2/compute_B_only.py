#!/bin/env python
"""Compute only the B matrix (outer product of gradients) locally.

B matrix is easier to compute - it's just the outer product of per-agent 
subgradients, doesn't require finite differences like A.

Run with: mpirun -n 4 python compute_B_only.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from bundlechoice import BundleChoice
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
NUM_SIMULS = 50  # Good balance
TIMELIMIT_SEC = 1.0

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
        "num_simulations": 1
    },
    "subproblem": {
        "name": "QuadKnapsack",
        "settings": {"TimeLimit": TIMELIMIT_SEC, "MIPGap_tol": 1e-2}
    },
    "standard_errors": {
        "num_simulations": NUM_SIMULS,
        "step_size": 1e-4,
        "seed": 1995,
    }
}

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

theta_hat = comm.bcast(theta_hat, root=0)

if rank == 0:
    print(f"\nMPI: {comm.Get_size()} ranks")
    print(f"Simulations: {NUM_SIMULS}")
    print(f"Computing ONLY B matrix...")

# Generate errors for SE
if rank == 0:
    print("\nGenerating error draws...")
    np.random.seed(1995)
    errors_all_sims = np.random.normal(0, 1, (NUM_SIMULS, num_agents, num_items))
else:
    errors_all_sims = None
errors_all_sims = comm.bcast(errors_all_sims, root=0)

# Cache observed features
if rank == 0:
    print("Caching observed features...")

local_obs_bundles = bc.data_manager.local_data["obs_bundles"]
obs_features_local = bc.feature_manager.compute_rank_features(local_obs_bundles)
obs_features_global = bc.comm_manager.concatenate_array_at_root_fast(obs_features_local, root=0)

if rank == 0:
    print(f"Observed features cached: shape={obs_features_global.shape}")

# Compute B matrix
if rank == 0:
    print(f"\nComputing B matrix ({num_features}×{num_features})...")

all_features_per_sim = []
for s in range(NUM_SIMULS):
    if rank == 0:
        print(f"  Simulation {s+1}/{NUM_SIMULS}...")
    
    # Update errors for this simulation
    bc.data.update_errors(errors_all_sims[s] if rank == 0 else None)
    bc.subproblems.initialize_local()
    
    # Solve subproblems
    local_num_agents = bc.data_manager.num_local_agents
    if local_num_agents > 0:
        local_bundles = bc.subproblem_manager.solve_local(theta_hat)
    else:
        local_bundles = np.empty((0, num_items), dtype=bool)
    
    # Gather features to root
    features_sim = bc.feature_manager.compute_gathered_features(local_bundles)
    if rank == 0:
        all_features_per_sim.append(features_sim)

comm.Barrier()

if rank == 0:
    # Stack: (S, N, K)
    features_all = np.stack(all_features_per_sim, axis=0)
    avg_simulated = features_all.mean(axis=0)  # (N, K)
    
    # Per-agent subgradient: g_i = avg_sim - obs
    g_i_full = avg_simulated - obs_features_global  # (N, K)
    
    # B = (1/N) sum_i g_i g_i^T
    B_global = (g_i_full.T @ g_i_full) / num_agents
    
    # Condition number
    B_cond = np.linalg.cond(B_global)
    
    print(f"\n{'='*60}")
    print("B MATRIX COMPUTED")
    print(f"{'='*60}")
    print(f"  Shape: {B_global.shape}")
    print(f"  Condition number: {B_cond:.2e}")
    print(f"  Diagonal range: [{B_global.diagonal().min():.4f}, {B_global.diagonal().max():.4f}]")
    print(f"  Diagonal mean: {B_global.diagonal().mean():.4f}")
    
    # Check for zeros on diagonal
    zero_diag = np.sum(B_global.diagonal() == 0)
    small_diag = np.sum(B_global.diagonal() < 1e-10)
    print(f"  Zero diagonal entries: {zero_diag}")
    print(f"  Near-zero diagonal entries (<1e-10): {small_diag}")
    
    # Save
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    np.save(os.path.join(OUTPUT_DIR, "B_matrix_local.npy"), B_global)
    print(f"\nSaved to: {OUTPUT_DIR}/B_matrix_local.npy")
    
    # Eigenvalue analysis
    eigvals = np.linalg.eigvalsh(B_global)
    print(f"\nEigenvalue analysis:")
    print(f"  Min eigenvalue: {eigvals.min():.6e}")
    print(f"  Max eigenvalue: {eigvals.max():.6e}")
    print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}")
    print(f"  Near-zero eigenvalues (<1e-10): {np.sum(np.abs(eigvals) < 1e-10)}")
    
    # Compare with saved B matrix from HPC
    try:
        B_hpc = np.load(os.path.join(OUTPUT_DIR, "sandwich_B_matrix.npy"))
        diff = np.abs(B_global - B_hpc)
        print(f"\nComparison with HPC B matrix:")
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")
        print(f"  HPC condition number: {np.linalg.cond(B_hpc):.2e}")
    except:
        print("\n(No HPC B matrix to compare)")
