#!/usr/bin/env python
"""
Example: Using custom feature oracles.

This shows how to define your own feature extraction function
instead of using auto-generation.

Run with: mpirun -n 10 python examples/02_custom_features.py
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Configuration
num_agents = 100
num_items = 20
num_features = 3  # Simple example: 2 linear + 1 quadratic
num_simuls = 1

# Define custom feature oracle
def custom_features(agent_id, bundle, data):
    """
    Custom feature extraction combining linear and quadratic terms.
    
    Args:
        agent_id: Index of the agent
        bundle: Binary array or matrix of bundles
        data: Dictionary containing agent and item data
    
    Returns:
        Feature vector of shape (num_features,) or (num_features, num_bundles)
    """
    agent_prefs = data["agent_data"]["preferences"][agent_id]
    
    # Handle both single bundle and multiple bundles
    if bundle.ndim == 1:
        # Single bundle
        linear_feat1 = agent_prefs[:, 0] @ bundle
        linear_feat2 = agent_prefs[:, 1] @ bundle
        quadratic_feat = -np.sum(bundle) ** 2  # Penalty for bundle size
        return np.array([linear_feat1, linear_feat2, quadratic_feat])
    else:
        # Multiple bundles (vectorized)
        linear_feat1 = agent_prefs[:, 0] @ bundle
        linear_feat2 = agent_prefs[:, 1] @ bundle
        quadratic_feat = -np.sum(bundle, axis=0) ** 2
        return np.vstack([linear_feat1, linear_feat2, quadratic_feat])

# Generate data on rank 0
if rank == 0:
    # Agent preferences: (num_agents, num_items, 2)
    agent_preferences = np.random.normal(0, 1, (num_agents, num_items, 2))
    errors = np.random.normal(0, 0.1, size=(num_simuls, num_agents, num_items))
    
    input_data = {
        "agent_data": {"preferences": agent_preferences},
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
    "ellipsoid": {"num_iters": 100, "verbose": False}
})

bc.data.load_and_scatter(input_data)

# Use custom feature oracle instead of build_from_data()
bc.features.set_oracle(custom_features)

# Generate observed bundles
theta_true = np.array([1.0, 0.5, 0.1])
obs_bundles = bc.subproblems.init_and_solve(theta_true)

if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.set_oracle(custom_features)
bc.subproblems.load()

# Estimate using ellipsoid method
theta_hat = bc.ellipsoid.solve()

if rank == 0:
    print("\n=== Custom Features Example ===")
    print(f"True theta:      {theta_true}")
    print(f"Estimated theta: {theta_hat}")
    print(f"Error:           {np.linalg.norm(theta_hat - theta_true):.4f}")
