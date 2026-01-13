#!/usr/bin/env python3
"""Test: Compare reset(0) vs no-reset on real Bayesian bootstrap."""

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

NUM_AGENTS = 200
NUM_ITEMS = 25
NUM_BOOT = 15
SEED = 42

if rank == 0:
    print("="*70)
    print("TEST: reset(0) vs no-reset on real Bayesian bootstrap")
    print("="*70)

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
    print(f"Constraints accumulated: {bc.row_generation.master_model.NumConstrs}")

# Generate same weights for both tests
np.random.seed(SEED)
all_weights = []
for _ in range(NUM_BOOT):
    w = np.random.exponential(1.0, NUM_AGENTS)
    all_weights.append(w / w.mean())

# ========== Test 1: Current approach (with reset(0)) ==========
if rank == 0:
    print(f"\n--- Method A: WITH reset(0) (current) ---")

# Re-run initial solve to reset state
bc.row_generation.solve()

t0 = time.time()
theta_boots_reset = []
iters_reset = []

for b, weights in enumerate(all_weights):
    weights_bcast = comm.bcast(weights if rank == 0 else None, root=0)
    result_b = bc.row_generation.solve_reuse_model(agent_weights=weights_bcast)
    
    if rank == 0:
        theta_boots_reset.append(result_b.theta_hat.copy())
        iters_reset.append(result_b.num_iterations)

time_reset = time.time() - t0

if rank == 0:
    se_reset = np.std(theta_boots_reset, axis=0, ddof=1)
    print(f"  Iterations: {iters_reset}")
    print(f"  Avg iters: {np.mean(iters_reset):.1f}, Time: {time_reset:.2f}s")
    print(f"  SE: {np.round(se_reset, 4)}")

# ========== Test 2: NO reset - just change Obj ==========
if rank == 0:
    print(f"\n--- Method B: WITHOUT reset (just change Obj) ---")

# Re-run initial solve to reset state  
bc.row_generation.solve()

t0 = time.time()
theta_boots_no_reset = []
iters_no_reset = []

for b, weights in enumerate(all_weights):
    if rank == 0:
        # Update objective manually without reset
        bc.row_generation.update_objective_for_weights(weights)
        # NO reset(0) - just optimize
        bc.row_generation.master_model.optimize()
        
        from gurobipy import GRB
        theta, u = bc.row_generation.master_variables
        if bc.row_generation.master_model.Status == GRB.OPTIMAL:
            bc.row_generation.theta_val = theta.X
        else:
            bc.row_generation.theta_val = np.zeros(4)
        
        iters = int(bc.row_generation.master_model.IterCount)
    else:
        bc.row_generation.theta_val = np.empty(4, dtype=np.float64)
        iters = 0
    
    bc.row_generation.theta_val = bc.comm_manager.broadcast_array(
        bc.row_generation.theta_val, root=0)
    
    # Run row generation iterations
    iteration = 0
    while iteration < bc.row_generation.row_generation_cfg.max_iters:
        local_pricing = bc.row_generation.subproblem_manager.solve_local(
            bc.row_generation.theta_val)
        stop = bc.row_generation._master_iteration(local_pricing)
        if stop:
            break
        iteration += 1
    
    if rank == 0:
        total_iters = iters + iteration + 1
        theta_boots_no_reset.append(bc.row_generation.theta_val.copy())
        iters_no_reset.append(total_iters)

time_no_reset = time.time() - t0

if rank == 0:
    se_no_reset = np.std(theta_boots_no_reset, axis=0, ddof=1)
    print(f"  Iterations: {iters_no_reset}")
    print(f"  Avg iters: {np.mean(iters_no_reset):.1f}, Time: {time_no_reset:.2f}s")
    print(f"  SE: {np.round(se_no_reset, 4)}")

# ========== Compare ==========
if rank == 0:
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"  With reset(0):    {np.mean(iters_reset):.1f} avg iters, {time_reset:.2f}s")
    print(f"  Without reset:    {np.mean(iters_no_reset):.1f} avg iters, {time_no_reset:.2f}s")
    
    # Check if results match
    se_diff = np.max(np.abs(se_reset - se_no_reset))
    print(f"\n  SE difference: {se_diff:.6f}")
    
    if se_diff < 0.01:
        print("  ✓ Results match!")
        if np.mean(iters_no_reset) < np.mean(iters_reset):
            speedup = np.mean(iters_reset) / np.mean(iters_no_reset)
            print(f"  ✓ No-reset is {speedup:.2f}x faster!")
        else:
            print("  ✗ No-reset is not faster")
    else:
        print("  ⚠ Results differ - need investigation")

comm.Barrier()
