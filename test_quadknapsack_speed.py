#!/usr/bin/env python3
"""Test quadknapsack solver speed with 1 agent and 200 items."""
import os
import numpy as np
from mpi4py import MPI
from bundlechoice.factory import ScenarioLibrary
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Test dimensions: 1 agent, 200 items
num_agents = 1
num_items = 200
num_agent_modular = 1
num_agent_quadratic = 1
num_item_modular = 1
num_item_quadratic = 1
num_simuls = 1
sigma = 1.0
seed = None  # Random seed (not fixed)

if rank == 0:
    print(f"Testing quadknapsack solver speed")
    print(f"  Agents: {num_agents}")
    print(f"  Items: {num_items}")
    print(f"  Agent modular features: {num_agent_modular}")
    print(f"  Agent quadratic features: {num_agent_quadratic}")
    print(f"  Item modular features: {num_item_modular}")
    print(f"  Item quadratic features: {num_item_quadratic}")
    print("=" * 80)

# Use factory to generate data (with Gurobi output enabled)
# Try different weight distributions for more heterogeneity
scenario = (
    ScenarioLibrary.quadratic_knapsack()
    .with_dimensions(num_agents=num_agents, num_items=num_items)
    .with_feature_counts(
        num_agent_modular=num_agent_modular,
        num_agent_quadratic=num_agent_quadratic,
        num_item_modular=num_item_modular,
        num_item_quadratic=num_item_quadratic,
    )
    .with_num_simuls(num_simuls)
    .with_sigma(sigma)
    .with_weight_config(
        distribution='lognormal',  # More heterogeneity: many small, few large
        low=1,
        high=100,  # Wider range for more heterogeneity
        log_mean=0.0,
        log_std=2.0,  # Higher std = more heterogeneity
    )
    .with_capacity_fraction(0.3)  # Standard: capacity = 30% of total weight
    .with_subproblem_settings(OutputFlag=1, TimeLimit=60)  # Enable Gurobi output, 1 minute timeout
    .build()
)

# Create custom theta
theta_0 = np.ones(4)  # 4 features

if rank == 0:
    print(f"\nTheta_0: {theta_0}")

# Prepare scenario with custom theta
# This generates bundles internally and returns them in estimation_data
if rank == 0:
    print("\nPreparing scenario and generating bundles...")
    tic = datetime.now()

prepared = scenario.prepare(comm=comm, seed=seed, theta=theta_0)

# Get observed bundles from prepare() (already computed with theta_0) - only on rank 0
if rank == 0:
    elapsed = (datetime.now() - tic).total_seconds()
    obs_bundles = prepared.estimation_data["obs_bundle"]
    print(f"✓ Bundle generation completed in {elapsed:.4f} seconds")
    print(f"\n  Aggregate demands: {obs_bundles.sum(1).min()} to {obs_bundles.sum(1).max()}")
    print(f"  Total aggregate: {obs_bundles.sum()}")
    print(f"  Mean demand per agent: {obs_bundles.sum(1).mean():.2f}")
    print("\n" + "=" * 80)
    print("Quadknapsack speed test completed successfully!")

