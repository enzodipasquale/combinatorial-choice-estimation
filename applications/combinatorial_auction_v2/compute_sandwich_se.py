#!/bin/env python
"""Compute sandwich standard errors for combinatorial auction estimation.

This script computes sandwich SEs by:
1. Computing per-agent subgradients g_i(theta) via simulation
2. Computing B matrix: (1/N) sum_i g_i g_i^T
3. Computing A matrix via finite differences
4. Computing sandwich variance: (1/N) A^{-1} B A^{-1}
"""

import numpy as np
from bundlechoice import BundleChoice
from mpi4py import MPI
import os
import pandas as pd
import time

# Monkey patch FeatureManager.compute_rank_features to handle ranks with 0 local agents
# (bundlechoice may error on empty local arrays when using many MPI ranks)
from bundlechoice.feature_manager import FeatureManager
_original_compute_rank_features = FeatureManager.compute_rank_features

def _patched_compute_rank_features(self, local_bundles):
    if getattr(self, "num_local_agents", 0) == 0 or local_bundles is None or len(local_bundles) == 0:
        return np.empty((0, self.num_features), dtype=np.float64)
    return _original_compute_rank_features(self, local_bundles)

FeatureManager.compute_rank_features = _patched_compute_rank_features

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
NUM_SIMULS_SE = int(os.environ.get("NUM_SIMULS_SE", "3"))
STEP_SIZE = float(os.environ.get("STEP_SIZE", "1e-4"))
TIMELIMIT_SEC = float(os.environ.get("TIMELIMIT_SEC", "1"))
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
SKIP_VALIDATION = os.environ.get("SKIP_VALIDATION", "0") == "1"

# USE_ALLREDUCE_MEAN is always enabled (optimized path using local sums + MPI Allreduce)
# This avoids gathering large per-agent arrays and is more scalable

# Override TimeLimit in TEST_MODE
if TEST_MODE:
    TIMELIMIT_SEC = 1.0
    if NUM_SIMULS_SE > 2:
        NUM_SIMULS_SE = 2

# Load theta_hat
if rank == 0:
    theta_hat = np.load(os.path.join(BASE_DIR, "estimation_results", "theta.npy"))
    print(f"Loaded theta_hat: shape={theta_hat.shape}")
else:
    theta_hat = None
theta_hat = comm.bcast(theta_hat, root=0)

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
    
    # Generate errors for all simulations
    np.random.seed(1995)
    errors_all_sims = np.random.normal(0, 1, size=(NUM_SIMULS_SE, num_agents, num_items))
    
    input_data = {
        "item_data": {"modular": -np.eye(num_items), "quadratic": quadratic, "weights": weights},
        "agent_data": {"modular": modular_agent, "capacity": capacity},
        "errors": errors_all_sims[0],  # Use first for initialization
        "obs_bundle": obs_bundle
    }
else:
    input_data = None
    num_agents = None
    num_items = None
    num_features = None
    errors_all_sims = None

num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
errors_all_sims = comm.bcast(errors_all_sims, root=0)

# Initialize BundleChoice
config = {
    "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                   "num_features": num_features, "num_simuls": 1},
    "subproblem": {"name": "QuadKnapsack", "settings": {"TimeLimit": TIMELIMIT_SEC, "MIPGap_tol": 1e-2}},
    "row_generation": {"max_iters": 400, "theta_ubs": 1000}
}

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

# Print test mode banner and rank distribution
if rank == 0:
    num_ranks = comm.Get_size()
    print("=" * 80)
    print("SANDBOX SE COMPUTATION - TEST MODE" if TEST_MODE else "SANDBOX SE COMPUTATION")
    print("=" * 80)
    print(f"MPI ranks: {num_ranks}")
    print(f"NUM_SIMULS_SE: {NUM_SIMULS_SE}")
    print(f"TimeLimit: {TIMELIMIT_SEC}s")
    print(f"SKIP_VALIDATION: {SKIP_VALIDATION}")
    print("=" * 80)

# Gather rank distribution info
num_local_agents_all = comm.allgather(bc.data_manager.num_local_agents)
if rank == 0:
    num_local_agents_array = np.array(num_local_agents_all)
    zero_agent_ranks = (num_local_agents_array == 0).sum()
    print(f"\nRank distribution:")
    print(f"  Total ranks: {len(num_local_agents_array)}")
    print(f"  Ranks with 0 agents: {zero_agent_ranks}")
    print(f"  Min local agents: {num_local_agents_array.min()}")
    print(f"  Mean local agents: {num_local_agents_array.mean():.2f}")
    print(f"  Max local agents: {num_local_agents_array.max()}")
    print("")

# Compute observed features
obs_bundles_local = bc.data_manager.local_data["obs_bundles"]
agents_obs_features = bc.feature_manager.compute_gathered_features(obs_bundles_local)

# Extract beta indices (non-FE parameters)
# Get dimensions from data_info (works on all ranks)
info = bc.data_manager.get_data_info()
num_modular_agent = info["num_modular_agent"]
num_modular_item = info["num_modular_item"]

if rank == 0:
    beta_indices = np.concatenate([
        np.arange(num_modular_agent),
        np.arange(num_modular_agent + num_modular_item, num_features)
    ]).astype(int)
    print(f"Beta indices (non-FE): {beta_indices}")
else:
    beta_indices = None
beta_indices = comm.bcast(beta_indices, root=0)

def compute_per_agent_subgradients(theta, errors_all_sims):
    """Compute g_i(theta) = (1/S) sum_s x_{i,B_i^{s,*}} - x_{i,B_i^obs}"""
    num_simuls = len(errors_all_sims)
    all_features_per_sim = []
    
    for s in range(num_simuls):
        if rank == 0:
            print(f"  Simulation {s+1}/{num_simuls}...", flush=True)
        bc.data_manager.update_errors(errors_all_sims[s] if rank == 0 else None)
        if bc.data_manager.num_local_agents > 0:
            local_bundles = bc.subproblems.solve_local(theta)
        else:
            local_bundles = np.empty((0, bc.data_manager.num_items), dtype=bool)
        features_sim = bc.feature_manager.compute_gathered_features(local_bundles)
        if rank == 0:
            all_features_per_sim.append(features_sim)
    
    if rank == 0:
        features_all = np.stack(all_features_per_sim, axis=0)  # (num_simuls, num_agents, num_features)
        avg_simulated = features_all.mean(axis=0)  # (num_agents, num_features)
        return avg_simulated - agents_obs_features  # (num_agents, num_features)
    return None

def compute_avg_subgradient_beta(theta, errors_all_sims):
    """Compute average subgradient g_bar^beta(theta).

    Uses local sums + MPI Allreduce to avoid gathering per-agent features to rank 0.
    This is valid because g_bar only needs the mean over agents, not the per-agent array.
    """
    num_simuls = len(errors_all_sims)
    num_beta = len(beta_indices)
    
    # Compute mean observed features (beta) via local sums + allreduce
    obs_local = bc.data_manager.local_data["obs_bundles"]
    obs_feat_local = bc.feature_manager.compute_rank_features(obs_local)  # (n_local, K)
    obs_sum_beta_local = obs_feat_local[:, beta_indices].sum(axis=0) if obs_feat_local.size else np.zeros(num_beta)

    obs_sum_beta_global = np.zeros(num_beta)
    comm.Allreduce(obs_sum_beta_local, obs_sum_beta_global, op=MPI.SUM)
    mean_obs_beta = obs_sum_beta_global / num_agents

    # Compute mean simulated features (beta) across simulations via local sums + allreduce
    sim_sum_beta_local = np.zeros(num_beta)

    for s in range(num_simuls):
        bc.data_manager.update_errors(errors_all_sims[s] if rank == 0 else None)
        
        if bc.data_manager.num_local_agents > 0:
            local_bundles = bc.subproblems.solve_local(theta)
        else:
            local_bundles = np.empty((0, bc.data_manager.num_items), dtype=bool)
        
        feat_local = bc.feature_manager.compute_rank_features(local_bundles)  # (n_local, K)
        if feat_local.size:
            sim_sum_beta_local += feat_local[:, beta_indices].sum(axis=0)

    sim_sum_beta_global = np.zeros(num_beta)
    comm.Allreduce(sim_sum_beta_local, sim_sum_beta_global, op=MPI.SUM)

    # Average over simulations, then over agents
    mean_sim_beta = (sim_sum_beta_global / num_simuls) / num_agents

    if rank == 0:
        return mean_sim_beta - mean_obs_beta
    return None

# Compute B matrix: (1/N) sum_i g_i^beta g_i^beta^T
if rank == 0:
    print(f"\n{'='*80}")
    print(f"Computing B matrix ({NUM_SIMULS_SE} simulations)...")
    print(f"{'='*80}")
g_i_theta_hat = compute_per_agent_subgradients(theta_hat, errors_all_sims)
if rank == 0:
    print(f"✓ B matrix computation complete")

if rank == 0:
    g_i_beta = g_i_theta_hat[:, beta_indices]  # (num_agents, num_beta)
    B_beta = (g_i_beta.T @ g_i_beta) / num_agents  # (num_beta, num_beta)
    print(f"B_beta: shape={B_beta.shape}, cond={np.linalg.cond(B_beta):.2e}")

# Compute A matrix via finite differences
# CRITICAL: All ranks must participate because compute_avg_subgradient_beta uses MPI Allreduce!
comm.Barrier()
if rank == 0:
    print(f"\n{'='*80}")
    print(f"Computing A matrix via finite differences (step_size={STEP_SIZE})...")
    print(f"{'='*80}")
num_beta = len(beta_indices)
A_beta = np.zeros((num_beta, num_beta)) if rank == 0 else None

for k_idx, k in enumerate(beta_indices):
    if rank == 0:
        print(f"  Column {k_idx+1}/{num_beta} (parameter index {k})...", flush=True)
    h_k = STEP_SIZE * max(1.0, abs(theta_hat[k]))
    theta_plus = theta_hat.copy()
    theta_plus[k] += h_k
    theta_minus = theta_hat.copy()
    theta_minus[k] -= h_k

    # NOTE: This is the dominant cost: 2 * (#simulations) * (#agents) MIP solves per column.
    # All ranks must call this function!
    g_plus = compute_avg_subgradient_beta(theta_plus, errors_all_sims)
    g_minus = compute_avg_subgradient_beta(theta_minus, errors_all_sims)
    
    if rank == 0:
        A_beta[:, k_idx] = (g_plus - g_minus) / (2 * h_k)
        if (k_idx + 1) % 10 == 0 or k_idx == len(beta_indices) - 1:
            print(f"    ✓ Completed {k_idx + 1}/{num_beta} columns", flush=True)

if rank == 0:
    print(f"✓ A matrix computation complete")
    print(f"A_beta: shape={A_beta.shape}, cond={np.linalg.cond(A_beta):.2e}")

# Compute sandwich variance: (1/N) A^{-1} B A^{-1}
comm.Barrier()
if rank == 0:
    print(f"\n{'='*80}")
    print("Computing sandwich variance...")
    print(f"{'='*80}")
    try:
        A_inv = np.linalg.solve(A_beta, np.eye(num_beta))
    except np.linalg.LinAlgError:
        print("Warning: A_beta singular, using pseudoinverse")
        A_inv = np.linalg.pinv(A_beta)
    
    V_beta = (1.0 / num_agents) * (A_inv @ B_beta @ A_inv.T)
    se_beta = np.sqrt(np.diag(V_beta))
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        "feature_index": beta_indices,
        "theta": theta_hat[beta_indices],
        "se": se_beta,
        "t_stat": theta_hat[beta_indices] / se_beta,
    })
    
    df.to_csv(os.path.join(OUTPUT_DIR, "sandwich_se.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIR, "sandwich_variance_beta.npy"), V_beta)
    
    print(f"\n✓ Standard errors saved to {OUTPUT_DIR}/sandwich_se.csv")
    print("\nResults:")
    for k_idx, k in enumerate(beta_indices):
        print(f"  Index {k}: theta={theta_hat[k]:.6f}, SE={se_beta[k_idx]:.6f}, t={theta_hat[k]/se_beta[k_idx]:.2f}")
    
    print(f"\n{'='*80}")
    print("DONE - Computation completed successfully")
    print(f"{'='*80}")
