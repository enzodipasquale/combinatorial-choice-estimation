#!/usr/bin/env python
"""
Small local test script for timing verification.
Uses small problem size (32 agents, 20 items, 2 simuls) - completes in seconds.
"""

import numpy as np
from bundlechoice import BundleChoice
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Small test problem (fast, completes in seconds)
if rank == 0:
    num_agents = 32
    num_items = 20
    num_features = 6
    num_simuls = 2
    
    np.random.seed(42)
    
    # Generate simple test data
    # Quadratic features must be non-negative with zero diagonal (upper triangular)
    quadratic = np.abs(np.random.randn(num_items, num_items, 2))
    # Make upper triangular and zero diagonal
    for k in range(2):
        quadratic[:, :, k] = np.triu(quadratic[:, :, k], k=1)
    
    # Feature breakdown: 1 agent modular + 1 item modular + 2 quadratic = 4 features
    # But config says 6, so let's use: 2 agent modular + 2 item modular + 2 quadratic = 6
    item_data = {
        "modular": np.random.randn(num_items, 2),  # 2 item modular features
        "quadratic": quadratic,  # 2 quadratic features
        "weights": np.ones(num_items)
    }
    agent_data = {
        "modular": np.random.randn(num_agents, num_items, 2),  # 2 agent modular features
        "capacity": np.ones(num_agents) * 5
    }
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    obs_bundle = np.random.rand(num_agents, num_items) > 0.5
    
    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    input_data = None
    num_agents = None
    num_items = None
    num_features = None
    num_simuls = None

# Broadcast dimensions
num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
num_simuls = comm.bcast(num_simuls, root=0)

# Configure and run
config = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls
    },
    "subproblem": {
        "name": "QuadSupermodularNetwork",
        "settings": {"TimeLimit": 5, "MIPGap_tol": 1e-2}
    },
    "row_generation": {
        "max_iters": 10,
        "tolerance_optimality": 0.01
    }
}

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# Run estimation
result = bc.row_generation.solve()

if rank == 0:
    print(f"\n{result.summary()}")

