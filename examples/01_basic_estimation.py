#!/usr/bin/env python
"""
Minimal example: Basic bundle choice estimation with greedy algorithm.

Run with: mpirun -n 10 python examples/01_basic_estimation.py
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Configuration
num_agents = 100
num_items = 20
num_features = 5
num_simuls = 1

# Generate data on rank 0
if rank == 0:
    # Simple linear features
    agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
    errors = np.random.normal(0, 0.1, size=(num_simuls, num_agents, num_items))
    
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

# Create and configure BundleChoice
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
        "max_iters": 50,
        "tolerance_optimality": 0.001,
        "gurobi_settings": {"OutputFlag": 0}
    }
})

# Load data
bc.data.load_and_scatter(input_data)

# Auto-generate feature oracle from data structure
bc.features.build_from_data()

# Generate observed bundles using true parameters
theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Add observed bundles to data for estimation
if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# Estimate parameters
theta_hat = bc.row_generation.solve()

if rank == 0:
    print("\n=== Results ===")
    print(f"True theta:      {theta_true}")
    print(f"Estimated theta: {theta_hat}")
    print(f"Error:           {np.linalg.norm(theta_hat - theta_true):.4f}")
