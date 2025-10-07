#!/usr/bin/env python
"""
Benchmark: Real estimation workflow to measure actual improvement.
Run with: mpirun -n 10 python benchmark_real_estimation.py
"""

import numpy as np
import time
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Configuration
num_agents = 1000
num_items = 30
num_features = 10
num_simuls = 1
max_iters = 10  # Quick benchmark

# Generate data on rank 0
if rank == 0:
    np.random.seed(42)
    agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
    errors = np.random.normal(0, 0.1, size=(num_simuls, num_agents, num_items))
    
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

# Setup BundleChoice
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
        "max_iters": max_iters,
        "min_iters": max_iters,  # Force exactly max_iters
        "tolerance_optimality": 1e-10,  # Very small, rarely stops early
        "gurobi_settings": {"OutputFlag": 0}
    }
})

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate observed bundles
theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

if rank == 0:
    input_data["obs_bundle"] = obs_bundles

# Reload with observations
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# Warm up
if rank == 0:
    print("Warming up...")
_ = bc.subproblems.solve_local(theta_true)
comm.Barrier()

# BENCHMARK
if rank == 0:
    print("\n" + "="*70)
    print(f"BASELINE BENCHMARK: Row Generation Estimation")
    print(f"  Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
    print(f"  Ranks: {comm.Get_size()}, Iterations: {max_iters}")
    print("="*70)

comm.Barrier()
t_start = time.time()

theta_hat = bc.row_generation.solve()

comm.Barrier()
t_total = time.time() - t_start

if rank == 0:
    print("\n" + "="*70)
    print(f"RESULTS:")
    print(f"  Total time: {t_total:.3f}s")
    print(f"  Time per iteration: {t_total/max_iters:.3f}s")
    print(f"  Parameter error: {np.linalg.norm(theta_hat - theta_true):.4f}")
    print("="*70)
    
    # Save baseline for comparison
    with open('baseline_time.txt', 'w') as f:
        f.write(f"{t_total:.6f}\n")
    print("\n✅ Baseline saved to baseline_time.txt")
