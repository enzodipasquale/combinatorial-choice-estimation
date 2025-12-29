#!/bin/env python
"""Minimal test to debug SE computation hang."""

import numpy as np
from bundlechoice import BundleChoice
from mpi4py import MPI
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create tiny test problem
if rank == 0:
    num_agents = 20
    num_items = 10
    num_features = 12
    
    np.random.seed(42)
    input_data = {
        "item_data": {
            "modular": -np.eye(num_items),
            "quadratic": np.random.randn(num_items, num_items, 2),
            "weights": np.ones(num_items)
        },
        "agent_data": {
            "modular": np.random.randn(num_agents, num_items, 1),
            "capacity": np.ones(num_agents) * 5
        },
        "errors": np.random.randn(num_agents, num_items),
        "obs_bundle": np.random.rand(num_agents, num_items) > 0.5
    }
    theta_hat = np.ones(num_features) * 0.1
    errors_all_sims = np.random.randn(1, num_agents, num_items)
else:
    input_data = None
    num_agents = None
    num_items = None
    num_features = None
    theta_hat = None
    errors_all_sims = None

num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
theta_hat = comm.bcast(theta_hat, root=0)
errors_all_sims = comm.bcast(errors_all_sims, root=0)

# Initialize
config = {
    "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                   "num_features": num_features, "num_simuls": 1},
    "subproblem": {"name": "QuadKnapsack", "settings": {"TimeLimit": 1, "MIPGap_tol": 1e-2}},
}

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

# Test compute_avg_subgradient_beta
if rank == 0:
    print(f"Testing compute_avg_subgradient_beta with {num_agents} agents, {comm.Get_size()} ranks")
    print(f"Rank 0 has {bc.data_manager.num_local_agents} local agents")

beta_indices = np.array([0, 10, 11])  # Simple beta indices

def test_compute_avg_subgradient_beta(theta, errors_all_sims):
    """Test version of compute_avg_subgradient_beta with timing."""
    num_simuls = len(errors_all_sims)
    num_beta = len(beta_indices)
    
    if rank == 0:
        print(f"  Step 1: Computing observed features...", flush=True)
    t0 = time.time()
    
    # Compute mean observed features
    obs_local = bc.data_manager.local_data["obs_bundles"]
    obs_feat_local = bc.feature_manager.compute_rank_features(obs_local)
    obs_sum_beta_local = obs_feat_local[:, beta_indices].sum(axis=0) if obs_feat_local.size else np.zeros(num_beta)
    
    if rank == 0:
        print(f"    Rank 0: obs_sum_beta_local = {obs_sum_beta_local}, time = {time.time()-t0:.2f}s", flush=True)
    
    t1 = time.time()
    obs_sum_beta_global = np.zeros(num_beta)
    comm.Allreduce(obs_sum_beta_local, obs_sum_beta_global, op=MPI.SUM)
    
    if rank == 0:
        print(f"  Step 2: Allreduce completed, time = {time.time()-t1:.2f}s", flush=True)
        print(f"    obs_sum_beta_global = {obs_sum_beta_global}", flush=True)
    
    mean_obs_beta = obs_sum_beta_global / num_agents
    
    # Compute mean simulated features
    sim_sum_beta_local = np.zeros(num_beta)
    
    for s in range(num_simuls):
        if rank == 0:
            print(f"  Step 3: Simulation {s+1}/{num_simuls} - updating errors...", flush=True)
        t2 = time.time()
        bc.data_manager.update_errors(errors_all_sims[s] if rank == 0 else None)
        if rank == 0:
            print(f"    Errors updated, time = {time.time()-t2:.2f}s", flush=True)
        
        if rank == 0:
            print(f"  Step 4: Solving subproblems...", flush=True)
        t3 = time.time()
        if bc.data_manager.num_local_agents > 0:
            local_bundles = bc.subproblems.solve_local(theta)
            if rank == 0:
                print(f"    Rank 0: solved {len(local_bundles)} bundles, time = {time.time()-t3:.2f}s", flush=True)
        else:
            local_bundles = np.empty((0, bc.data_manager.num_items), dtype=bool)
            if rank == 0:
                print(f"    Rank 0: no local agents, time = {time.time()-t3:.2f}s", flush=True)
        
        if rank == 0:
            print(f"  Step 5: Computing features...", flush=True)
        t4 = time.time()
        feat_local = bc.feature_manager.compute_rank_features(local_bundles)
        if rank == 0:
            print(f"    Rank 0: features computed, shape = {feat_local.shape}, time = {time.time()-t4:.2f}s", flush=True)
        
        if feat_local.size:
            sim_sum_beta_local += feat_local[:, beta_indices].sum(axis=0)
            if rank == 0:
                print(f"    Rank 0: sim_sum_beta_local = {sim_sum_beta_local}", flush=True)
    
    if rank == 0:
        print(f"  Step 6: Final Allreduce...", flush=True)
    t5 = time.time()
    sim_sum_beta_global = np.zeros(num_beta)
    comm.Allreduce(sim_sum_beta_local, sim_sum_beta_global, op=MPI.SUM)
    
    if rank == 0:
        print(f"    Allreduce completed, time = {time.time()-t5:.2f}s", flush=True)
        print(f"    sim_sum_beta_global = {sim_sum_beta_global}", flush=True)
    
    mean_sim_beta = (sim_sum_beta_global / num_simuls) / num_agents
    
    if rank == 0:
        result = mean_sim_beta - mean_obs_beta
        print(f"  Result: {result}", flush=True)
        return result
    return None

# Test it
if rank == 0:
    print("\n" + "="*60)
    print("TESTING compute_avg_subgradient_beta")
    print("="*60)
result = test_compute_avg_subgradient_beta(theta_hat, errors_all_sims)

if rank == 0:
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

