#!/usr/bin/env python3
"""Compare quadknapsack solver v1 vs v2 performance."""
import os
import numpy as np
from mpi4py import MPI
from bundlechoice.factory import ScenarioLibrary
from datetime import datetime
import time

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
seed = 12345  # Fixed seed for reproducibility

if rank == 0:
    print("=" * 80)
    print("Quadratic Knapsack Solver Comparison: v1 vs v2")
    print("=" * 80)
    print(f"  Agents: {num_agents}")
    print(f"  Items: {num_items}")
    print(f"  Agent modular features: {num_agent_modular}")
    print(f"  Agent quadratic features: {num_agent_quadratic}")
    print(f"  Item modular features: {num_item_modular}")
    print(f"  Item quadratic features: {num_item_quadratic}")
    print("=" * 80)

# Create custom theta
theta_0 = np.ones(4)  # 4 features

if rank == 0:
    print(f"\nTheta_0: {theta_0}\n")

# Test both versions
results = {}

for version_name, subproblem_name in [("v1", "QuadKnapsack"), ("v2", "QuadKnapsackV2")]:
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Testing {version_name.upper()} ({subproblem_name})")
        print(f"{'='*80}")
    
    # Create scenario with specific subproblem
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
            distribution='uniform',
            low=1,
            high=100,
        )
        .with_subproblem_settings(OutputFlag=0, TimeLimit=60)  # Disable output for timing
        .build()
    )
    
    # Override subproblem name in config factory for v2
    if version_name == "v2":
        original_config_factory = scenario.config_factory
        def v2_config_factory():
            config = original_config_factory()
            config["subproblem"]["name"] = subproblem_name
            return config
        scenario.config_factory = v2_config_factory
    
    # Prepare scenario and time it
    if rank == 0:
        print(f"Preparing scenario and generating bundles...")
        tic = time.perf_counter()
    
    prepared = scenario.prepare(comm=comm, seed=seed, theta=theta_0)
    
    if rank == 0:
        elapsed = time.perf_counter() - tic
        obs_bundles = prepared.estimation_data["obs_bundle"]
        results[version_name] = {
            'time': elapsed,
            'bundles': obs_bundles,
            'total_demand': obs_bundles.sum(),
            'mean_demand': obs_bundles.sum(1).mean(),
        }
        print(f"✓ {version_name.upper()} completed in {elapsed:.4f} seconds")
        print(f"  Total aggregate demand: {obs_bundles.sum()}")
        print(f"  Mean demand per agent: {obs_bundles.sum(1).mean():.2f}")

# Compare results
if rank == 0:
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    v1_time = results['v1']['time']
    v2_time = results['v2']['time']
    speedup = v1_time / v2_time if v2_time > 0 else float('inf')
    
    print(f"v1 time: {v1_time:.4f} seconds")
    print(f"v2 time: {v2_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✓ v2 is {speedup:.2f}x FASTER")
    elif speedup < 1.0:
        print(f"✗ v2 is {1/speedup:.2f}x SLOWER")
    else:
        print(f"≈ Performance is similar")
    
    # Verify solutions are the same (or similar)
    v1_bundles = results['v1']['bundles']
    v2_bundles = results['v2']['bundles']
    
    print(f"\n{'='*80}")
    print("SOLUTION COMPARISON")
    print(f"{'='*80}")
    print(f"v1 total demand: {results['v1']['total_demand']}")
    print(f"v2 total demand: {results['v2']['total_demand']}")
    
    if np.array_equal(v1_bundles, v2_bundles):
        print("✓ Solutions are IDENTICAL")
    else:
        diff = np.abs(v1_bundles.astype(int) - v2_bundles.astype(int)).sum()
        print(f"⚠ Solutions differ in {diff} bundle positions")
        print(f"  (This may be expected if multiple optimal solutions exist)")
    
    print(f"\n{'='*80}")
    print("Comparison completed!")
    print(f"{'='*80}")

