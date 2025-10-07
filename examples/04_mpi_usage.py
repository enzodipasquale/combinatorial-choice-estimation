#!/usr/bin/env python
"""
Example: MPI-specific patterns and best practices.

This demonstrates how to properly use BundleChoice in an MPI environment,
including data distribution, rank-specific operations, and validation.

Run with: mpirun -n 10 python examples/04_mpi_usage.py
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}/{size}] Starting")

# Configuration
num_agents = 200
num_items = 30
num_features = 5
num_simuls = 1

# ===== DATA GENERATION (RANK 0 ONLY) =====
if rank == 0:
    print(f"\n[Rank 0] Generating data for {num_agents} agents...")
    
    agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
    errors = np.random.normal(0, 0.1, size=(num_simuls, num_agents, num_items))
    
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
    
    print(f"[Rank 0] Data generated. Shape: {agent_features.shape}")
else:
    # Other ranks don't need data - it will be scattered
    input_data = None

# ===== SETUP BUNDLECHOICE (ALL RANKS) =====
bc = BundleChoice()
bc.load_config({
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls
    },
    "subproblem": {"name": "Greedy"},
    "row_generation": {
        "max_iters": 30,
        "tolerance_optimality": 0.001,
        "gurobi_settings": {"OutputFlag": 0}
    }
})

# ===== DATA DISTRIBUTION =====
# Data is scattered from rank 0 to all ranks
bc.data.load_and_scatter(input_data)

print(f"[Rank {rank}] Received {bc.data.num_local_agents} local agents")

# ===== FEATURE SETUP (ALL RANKS) =====
bc.features.build_from_data()

# ===== VALIDATE SETUP =====
# Use validate_setup() to check configuration before running
try:
    bc.validate_setup('row_generation')
    if rank == 0:
        print("\n[Rank 0] ✅ Setup validation passed")
except RuntimeError as e:
    if rank == 0:
        print(f"\n[Rank 0] ❌ Setup validation failed: {e}")
    comm.Abort(1)

# ===== GENERATE OBSERVED BUNDLES (DISTRIBUTED) =====
if rank == 0:
    print("\n[Rank 0] Generating observed bundles...")

theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Only rank 0 gets the full result
if rank == 0:
    print(f"[Rank 0] Generated {obs_bundles.shape[0]} observed bundles")
    input_data["obs_bundle"] = obs_bundles

# ===== RELOAD DATA WITH OBSERVATIONS =====
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# ===== ESTIMATION (DISTRIBUTED) =====
if rank == 0:
    print("\n[Rank 0] Starting parameter estimation...")
    print(f"[Rank 0] Each of {size} ranks will solve ~{num_agents//size} subproblems per iteration")

# Define callback to monitor progress (rank 0 only)
def progress_callback(info):
    if info['iteration'] % 5 == 0:
        print(f"[Rank 0] Iteration {info['iteration']:3d}: "
              f"obj={info['objective']:8.2f}, "
              f"pricing={info['pricing_time']:.3f}s, "
              f"master={info['master_time']:.3f}s")

theta_hat = bc.row_generation.solve(callback=progress_callback)

# ===== RESULTS (RANK 0 ONLY) =====
if rank == 0:
    print("\n=== MPI Example Results ===")
    print(f"True theta:      {theta_true}")
    print(f"Estimated theta: {theta_hat}")
    print(f"Error:           {np.linalg.norm(theta_hat - theta_true):.4f}")
    
    # Get subproblem statistics
    stats = bc.subproblems.get_stats()
    print(f"\nSubproblem statistics:")
    print(f"  Total solves: {stats['num_solves']}")
    print(f"  Total time:   {stats['total_time']:.2f}s")
    print(f"  Mean time:    {stats['mean_time']:.4f}s")
    print(f"  Max time:     {stats['max_time']:.4f}s")

print(f"[Rank {rank}] Finished")
