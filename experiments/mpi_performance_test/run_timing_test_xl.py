#!/bin/env python
"""
XL timing test: Compare main vs feature branch timing fixes
"""

from bundlechoice import BundleChoice
from bundlechoice.factory import ScenarioLibrary
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# XL test parameters
num_agents = 512
num_items = 200
num_modular_agent_features = 2
num_modular_item_features = 2
num_quadratic_item_features = 2
num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_item_features
num_simuls = 5
sigma = 5.0

if rank == 0:
    print("=" * 70)
    print("TIMING TEST - SUPERMODULAR ESTIMATION (XL)")
    print("=" * 70)
    print(f"Problem: {num_agents} agents × {num_items} items, {num_features} features")
    print(f"Simulations: {num_simuls}")
    print(f"MPI Ranks: {comm.Get_size()}")
    print("=" * 70)
    print()
    sys.stdout.flush()

# Generate data using factory
theta_star = np.ones(num_features) * 2.0

scenario = (
    ScenarioLibrary.quadratic_supermodular()
    .with_dimensions(num_agents=num_agents, num_items=num_items)
    .with_feature_counts(
        num_mod_agent=num_modular_agent_features,
        num_mod_item=num_modular_item_features,
        num_quad_item=num_quadratic_item_features,
    )
    .with_sigma(sigma)
    .with_num_simuls(num_simuls)
    .build()
)

# Prepare scenario (generates data and observed bundles)
prepared = scenario.prepare(comm=comm, seed=42, theta=theta_star)

# Extract estimation data
estimation_data = prepared.estimation_data

# Update config
config = prepared.config.copy()
config["dimensions"]["num_simuls"] = num_simuls
config["row_generation"]["max_iters"] = 100  # Limit iterations for quick test

# Initialize BundleChoice and load data
if rank == 0:
    print("Initializing BundleChoice...")
    sys.stdout.flush()

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(estimation_data)
bc.features.build_from_data()
bc.subproblems.load()

# Run estimation
if rank == 0:
    print("Starting row generation estimation...")
    print()
    sys.stdout.flush()

theta_hat = bc.row_generation.solve()

# Results are printed by row_generation.solve()
if rank == 0:
    sys.stdout.flush()



