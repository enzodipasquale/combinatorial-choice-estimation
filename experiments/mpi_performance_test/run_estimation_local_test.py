#!/bin/env python
"""
Quick Local Test: Small problem to verify timing fixes work
Fast test for local agent - should complete in seconds, not minutes
"""

from bundlechoice import BundleChoice
from bundlechoice.factory import ScenarioLibrary
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# SMALL test parameters - fast to solve
num_agents = 32   # Small (was 512)
num_items = 20    # Small (was 200)
num_modular_agent_features = 2
num_modular_item_features = 2
num_quadratic_item_features = 2
num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_item_features
num_simuls = 2    # Small (was 10)
sigma = 5.0

if rank == 0:
    print("=" * 70)
    print("LOCAL QUICK TEST - TIMING VERIFICATION")
    print("=" * 70)
    print(f"Problem Dimensions:")
    print(f"  • Agents: {num_agents}")
    print(f"  • Items: {num_items}")
    print(f"  • Features: {num_features}")
    print(f"  • Simulations: {num_simuls}")
    print(f"  • MPI Ranks: {comm.Get_size()}")
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

# Update config to have correct num_simuls for estimation
config = prepared.config.copy()
config["dimensions"]["num_simuls"] = num_simuls

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

# Print results
if rank == 0:
    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    if theta_hat is not None:
        theta_array = theta_hat.theta_hat if hasattr(theta_hat, 'theta_hat') else theta_hat
        print(f"Theta estimate shape: {theta_array.shape}")
    
    # Print enhanced diagnostics if available
    if hasattr(bc.row_generation, 'timing_stats') and bc.row_generation.timing_stats:
        stats = bc.row_generation.timing_stats
        print()
        print("=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        print(f"Total runtime: {stats.get('total_time', 0):.2f}s")
        print(f"MPI gather time: {stats.get('mpi_time', 0):.2f}s ({stats.get('mpi_time_pct', 0):.1f}%)")
        print()
        print("Note: Full timing breakdown should appear in 'Timing Statistics:' section above.")
        print("=" * 70)
    
    sys.stdout.flush()

