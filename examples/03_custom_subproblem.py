#!/usr/bin/env python
"""
Example: Implementing a custom subproblem solver.

This shows how researchers can add their own optimization algorithms
by inheriting from SerialSubproblemBase or BatchSubproblemBase.

Run with: mpirun -n 10 python examples/03_custom_subproblem.py
"""

import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.subproblems.base import SerialSubproblemBase

class RandomSearchSubproblem(SerialSubproblemBase):
    """
    Custom subproblem using random search (for demonstration).
    In practice, you'd implement a more sophisticated algorithm.
    """
    
    def initialize(self, local_id):
        """Initialize problem for agent local_id."""
        # No initialization needed for random search
        return None
    
    def solve(self, local_id, theta, pb=None):
        """
        Solve subproblem using random search.
        
        Args:
            local_id: Local agent ID
            theta: Parameter vector
            pb: Problem state (unused)
        
        Returns:
            Best bundle found
        """
        error_j = self.local_data["errors"][local_id]
        
        # Random search with 100 samples
        num_samples = 100
        best_value = -np.inf
        best_bundle = np.zeros(self.num_items, dtype=bool)
        
        for _ in range(num_samples):
            # Random bundle
            bundle = np.random.rand(self.num_items) > 0.5
            
            # Compute value
            features = self.features_oracle(local_id, bundle, self.local_data)
            value = features @ theta + error_j @ bundle
            
            if value > best_value:
                best_value = value
                best_bundle = bundle.copy()
        
        return best_bundle

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Configuration
num_agents = 50
num_items = 10
num_features = 3
num_simuls = 1

# Generate data
if rank == 0:
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
    "subproblem": {"name": "Greedy"},  # Used for generating obs_bundles
    "ellipsoid": {"num_iters": 50, "verbose": False}
})

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate observed bundles using Greedy
theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Use custom subproblem solver
bc.subproblems.load(RandomSearchSubproblem)

# Estimate parameters
theta_hat = bc.ellipsoid.solve()

if rank == 0:
    print("\n=== Custom Subproblem Example ===")
    print(f"Using custom RandomSearchSubproblem")
    print(f"True theta:      {theta_true}")
    print(f"Estimated theta: {theta_hat}")
    print(f"Error:           {np.linalg.norm(theta_hat - theta_true):.4f}")
