#!/usr/bin/env python3
"""Final test: Verify Bayesian bootstrap works after removing reset(0)."""

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

NUM_AGENTS = 150
NUM_ITEMS = 20
NUM_BOOT = 10
SEED = 42

if rank == 0:
    print("="*60)
    print("FINAL TEST: Bayesian bootstrap without reset(0)")
    print("="*60)

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

# Initial solve
result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)

if rank == 0:
    print(f"\nInitial: {result.num_iterations} iters")
    print(f"theta_hat = {np.round(theta_hat, 3)}")
    print(f"Constraints: {bc.row_generation.master_model.NumConstrs}")

# Bayesian bootstrap
np.random.seed(SEED)
t0 = time.time()
theta_boots = []
iters_list = []

for b in range(NUM_BOOT):
    if rank == 0:
        weights = np.random.exponential(1.0, NUM_AGENTS)
        weights = weights / weights.mean()
    else:
        weights = None
    
    result_b = bc.row_generation.solve_reuse_model(agent_weights=weights)
    
    if rank == 0:
        theta_boots.append(result_b.theta_hat.copy())
        iters_list.append(result_b.num_iterations)

elapsed = time.time() - t0

if rank == 0:
    theta_boots = np.array(theta_boots)
    se = np.std(theta_boots, axis=0, ddof=1)
    
    print(f"\nBootstrap completed:")
    print(f"  Iterations: {iters_list}")
    print(f"  Avg iters: {np.mean(iters_list):.1f}")
    print(f"  Time: {elapsed:.2f}s ({elapsed/NUM_BOOT*1000:.1f}ms per sample)")
    print(f"\n  SE: {np.round(se, 4)}")
    
    # Verify results
    all_different = len(set(map(tuple, theta_boots))) == NUM_BOOT
    se_nonzero = np.all(se > 0)
    
    if all_different and se_nonzero:
        print("\n✓ SUCCESS: All bootstrap samples different, SEs non-zero")
    else:
        print("\n✗ FAIL: Check results!")
        if not all_different:
            print("  - Some bootstrap samples are identical (cached?)")
        if not se_nonzero:
            print("  - Some SEs are zero")

comm.Barrier()
