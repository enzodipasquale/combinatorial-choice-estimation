#!/usr/bin/env python3
"""Quick test: Verify improvements don't break Bayesian bootstrap."""

import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys, os, logging
if rank != 0:
    sys.stdout = open(os.devnull, 'w')
logging.disable(logging.WARNING)

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

NUM_AGENTS = 80
NUM_ITEMS = 12
NUM_BOOT = 5
SEED = 42

if rank == 0:
    print("="*60)
    print("TEST: Verify Bayesian bootstrap after improvements")
    print("="*60)

# Setup
scenario = (ScenarioLibrary.linear_knapsack()
    .with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS)
    .with_feature_counts(num_agent_features=2, num_item_features=2)
    .build())

prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
config = {
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, 
                   "num_features": 4, "num_simulations": 1},
    "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 1.0}},
    "row_generation": {"max_iters": 50, "theta_ubs": 100},
}
bc.load_config(config)
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

# Initial estimation
result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)

if rank == 0:
    print(f"\nInitial: {result.num_iterations} iters, theta={np.round(theta_hat, 3)}")

# Bayesian bootstrap
np.random.seed(SEED)
t0 = time.time()
theta_boots = []

for b in range(NUM_BOOT):
    if rank == 0:
        weights = np.random.exponential(1.0, NUM_AGENTS)
        weights = weights / weights.mean()
    else:
        weights = None
    
    result_b = bc.row_generation.solve_reuse_model(agent_weights=weights)
    
    if rank == 0:
        theta_boots.append(result_b.theta_hat.copy())
        print(f"  Boot {b+1}: {result_b.num_iterations} iters, theta={np.round(result_b.theta_hat, 3)}")

if rank == 0:
    elapsed = time.time() - t0
    theta_boots = np.array(theta_boots)
    se = np.std(theta_boots, axis=0, ddof=1)
    
    print(f"\nBootstrap SE: {np.round(se, 4)}")
    print(f"Time: {elapsed:.2f}s for {NUM_BOOT} samples")
    
    # Verify SEs are non-zero (not returning cached solutions)
    if np.all(se > 0):
        print("\n✓ SUCCESS: SEs are non-zero, bootstrap working correctly")
    else:
        print("\n✗ FAIL: Some SEs are zero!")

comm.Barrier()
if rank == 0:
    print("\nDone.")
