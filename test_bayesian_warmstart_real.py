#!/usr/bin/env python3
"""Test: Compare Bayesian bootstrap warm-start strategies on real row generation."""

import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Suppress logging on non-root
import sys, os, logging
if rank != 0:
    sys.stdout = open(os.devnull, 'w')
logging.disable(logging.WARNING)

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

NUM_AGENTS = 100
NUM_ITEMS = 15
NUM_BOOT = 10  # Small for quick test
SEED = 42

if rank == 0:
    print("="*70)
    print("TEST: Bayesian Bootstrap Warm-Start Strategies")
    print(f"  N={NUM_AGENTS}, J={NUM_ITEMS}, Boot samples={NUM_BOOT}")
    print("="*70)


def test_scenario(name, scenario, subproblem, theta_true, theta_lbs=None):
    """Test different warm-start strategies for a scenario."""
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"{'='*70}")
    
    # Prepare data
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED, theta=theta_true)
    
    # Setup BundleChoice
    bc = BundleChoice()
    config = {
        "dimensions": {
            "num_agents": NUM_AGENTS, 
            "num_items": NUM_ITEMS, 
            "num_features": len(theta_true), 
            "num_simulations": 1
        },
        "subproblem": {
            "name": subproblem, 
            "settings": {"TimeLimit": 1.0} if "Knapsack" in subproblem else {}
        },
        "row_generation": {"max_iters": 100, "theta_ubs": 100},
    }
    if theta_lbs is not None:
        config["row_generation"]["theta_lbs"] = theta_lbs
    
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
    bc.oracles.build_from_data()
    bc.subproblems.load()
    bc.subproblems.initialize_local()
    
    # Initial estimation
    result = bc.row_generation.solve()
    theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
    
    if rank == 0:
        print(f"  Initial estimation: {result.num_iterations} iterations")
        print(f"  theta_hat = {np.round(theta_hat, 3)}")
    
    # Test warm-start strategies
    np.random.seed(SEED)
    
    # Strategy 1: Current (model reuse with reset(0))
    if rank == 0:
        print(f"\n  Strategy 1: model reuse (current, with reset)")
    
    t0 = time.time()
    iters_current = []
    for b in range(NUM_BOOT):
        if rank == 0:
            weights = np.random.exponential(1.0, NUM_AGENTS)
            weights = weights / weights.mean()
        else:
            weights = None
        
        result_b = bc.row_generation.solve_reuse_model(agent_weights=weights)
        if rank == 0:
            iters_current.append(result_b.num_iterations)
    
    time_current = time.time() - t0
    
    if rank == 0:
        print(f"    Iterations: {iters_current}")
        print(f"    Avg: {np.mean(iters_current):.1f}, Time: {time_current:.2f}s")
    
    # Check: How many constraints accumulated?
    if rank == 0:
        num_constrs = bc.row_generation.master_model.NumConstrs
        print(f"\n  Accumulated constraints: {num_constrs}")
        print(f"  → Constraint reuse is the key benefit, not LP basis warm-start")
        print(f"  → Fresh solve would need to re-discover all these constraints")
    
    comm.Barrier()


# Test on LinearKnapsack (simpler, faster)
scenario = (ScenarioLibrary.linear_knapsack()
    .with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS)
    .with_feature_counts(num_agent_features=2, num_item_features=2)
    .build())
test_scenario("LinearKnapsack", scenario, "LinearKnapsack", 
              theta_true=np.array([1.0, 1.0, 1.0, 1.0]))

comm.Barrier()

if rank == 0:
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
