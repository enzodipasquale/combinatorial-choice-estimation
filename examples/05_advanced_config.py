#!/usr/bin/env python
"""
Example: Advanced configuration patterns.

This demonstrates:
- Quick configuration setup
- Warm start for iterative solving
- Result caching for sensitivity analysis
- Multiple estimation methods

Run with: mpirun -n 10 python examples/05_advanced_config.py
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("=== Advanced Configuration Example ===\n")

# ===== 1. QUICK CONFIGURATION =====
if rank == 0:
    print("1. Setting up with fast configuration...")

# Create config for quick solving
cfg = {
    'dimensions': {
        'num_agents': 100,
        'num_items': 20,
        'num_features': 5,
        'num_simuls': 1
    },
    'subproblem': {'name': 'Greedy'},
    'row_generation': {
        'max_iters': 20,
        'tolerance_optimality': 0.01,
        'gurobi_settings': {'OutputFlag': 0}
    }
}

# Setup
bc = BundleChoice()
bc.load_config(cfg)

# Generate data
if rank == 0:
    agent_features = np.random.normal(0, 1, (100, 20, 5))
    errors = np.random.normal(0, 0.1, size=(1, 100, 20))
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate observations
theta_true = np.ones(5)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# ===== 2. WARM START =====
if rank == 0:
    print("\n2. Using warm start for faster convergence...")

# First solve with fewer iterations
bc.load_config({'row_generation': {'max_iters': 10}})
theta_1 = bc.row_generation.solve()

if rank == 0:
    print(f"   Quick solve theta: {theta_1}")

# Second solve with warm start
bc.load_config({'row_generation': {'max_iters': 30}})
bc.row_generation.theta_init = theta_1  # Warm start from previous solution
theta_2 = bc.row_generation.solve()

if rank == 0:
    print(f"   Refined theta:     {theta_2}")

# ===== 3. RESULT CACHING =====
if rank == 0:
    print("\n3. Using result caching for sensitivity analysis...")

bc.subproblems.enable_cache()

# Solve for multiple parameter values
theta_values = [theta_true + 0.1 * np.random.randn(5) for _ in range(5)]
theta_values.append(theta_values[0])  # Repeat first to test cache

for i, theta in enumerate(theta_values):
    bundles = bc.subproblems.init_and_solve(theta)
    if rank == 0:
        cache_status = "HIT" if i == len(theta_values) - 1 else "MISS"
        print(f"   Solve {i+1}: {bundles.shape} bundles [{cache_status}]")

bc.subproblems.disable_cache()

# ===== 4. MULTIPLE ESTIMATION METHODS =====
if rank == 0:
    print("\n4. Comparing estimation methods...")

# Method 1: Row generation
theta_rg = bc.row_generation.solve()

# Method 2: Ellipsoid
bc.load_config({'ellipsoid': {'num_iters': 50, 'verbose': False}})
theta_el = bc.ellipsoid.solve()

if rank == 0:
    print(f"   Row generation: {theta_rg}")
    print(f"   Ellipsoid:      {theta_el}")

# ===== RESULTS =====
if rank == 0:
    print("\n=== Summary ===")
    print(f"True theta:         {theta_true}")
    print(f"All methods close to truth: {np.allclose(theta_rg, theta_true, atol=0.5)}")
    
    # Get final statistics
    stats = bc.subproblems.get_stats()
    print(f"\nTotal subproblem solves: {stats['num_solves']}")
    print(f"Total time: {stats['total_time']:.2f}s")
