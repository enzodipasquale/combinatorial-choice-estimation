#!/usr/bin/env python3
"""Benchmark: Compare model vs model_reset on 5 standard settings."""

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

NUM_AGENTS = 100
NUM_ITEMS = 15
NUM_BOOT = 15
SEED = 42

if rank == 0:
    print("="*70)
    print("BENCHMARK: model (LP warm) vs model_reset (cold)")
    print(f"  N={NUM_AGENTS}, J={NUM_ITEMS}, Bootstrap={NUM_BOOT}")
    print("="*70)


def greedy_oracle(agent_idx, bundles, data):
    modular = data["agent_data"]["modular"][agent_idx]
    modular = np.atleast_2d(modular)
    single = bundles.ndim == 1
    if single:
        bundles = bundles[:, None]
    modular_feat = modular.T @ bundles
    quad_feat = -np.sum(bundles, axis=0, keepdims=True) ** 2
    features = np.vstack((modular_feat, quad_feat))
    return features[:, 0] if single else features


def setup_bc(scenario, subproblem, theta_true, use_greedy=False, theta_lbs=None):
    """Setup fresh BundleChoice."""
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED, theta=theta_true)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, 
                       "num_features": len(theta_true), "num_simulations": 1},
        "subproblem": {"name": subproblem, 
                       "settings": {"TimeLimit": 0.3} if "Knapsack" in subproblem else {}},
        "row_generation": {"max_iters": 50, "theta_ubs": 100},
    }
    if theta_lbs:
        config["row_generation"]["theta_lbs"] = theta_lbs
    
    bc.load_config(config)
    bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
    
    if use_greedy:
        bc.oracles.set_features_oracle(greedy_oracle)
        bc.subproblems.load()
        from bundlechoice.subproblems.registry.greedy import GreedySubproblem
        from bundlechoice.scenarios.greedy import _install_find_best_item
        if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
            _install_find_best_item(bc.subproblems.subproblem_instance)
    else:
        bc.oracles.build_from_data()
        bc.subproblems.load()
    bc.subproblems.initialize_local()
    
    return bc, prepared


def run_single_warmstart(name, scenario, subproblem, theta_true, warmstart, 
                         use_greedy=False, theta_lbs=None):
    """Run bootstrap with single warmstart strategy."""
    bc, prepared = setup_bc(scenario, subproblem, theta_true, use_greedy, theta_lbs)
    
    result = bc.row_generation.solve()
    theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
    
    np.random.seed(SEED)
    t0 = time.time()
    
    se_result = bc.standard_errors.compute_bayesian_bootstrap(
        theta_hat=theta_hat,
        row_generation=bc.row_generation,
        num_bootstrap=NUM_BOOT,
        seed=SEED,
        warmstart=warmstart,
    )
    
    elapsed = time.time() - t0
    se = se_result.se if se_result else None
    
    return elapsed, se


def benchmark(name, scenario, subproblem, theta_true, use_greedy=False, theta_lbs=None):
    """Benchmark both strategies."""
    if rank == 0:
        print(f"\n{name}")
    
    # Run model (LP warm-start)
    t1, se1 = run_single_warmstart(name, scenario, subproblem, theta_true, "model", 
                                    use_greedy, theta_lbs)
    comm.Barrier()
    
    # Run model_reset (cold start)
    t2, se2 = run_single_warmstart(name, scenario, subproblem, theta_true, "model_reset",
                                    use_greedy, theta_lbs)
    comm.Barrier()
    
    if rank == 0:
        speedup = t2 / t1 if t1 > 0 else 1.0
        match = np.allclose(se1, se2, atol=0.01) if se1 is not None and se2 is not None else False
        print(f"  model={t1:.3f}s, reset={t2:.3f}s, speedup={speedup:.2f}x {'✓' if speedup>1.05 else ''}, SE {'✓' if match else '⚠'}")
        return {'name': name, 't_model': t1, 't_reset': t2, 'speedup': speedup}
    return None


results = []

# Run benchmarks
scenarios = [
    ("1.Greedy", ScenarioLibrary.greedy().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_num_features(3).build(), 
     "Greedy", np.array([1.0, 1.0, 0.1]), True, None),
    ("2.LinearKnapsack", ScenarioLibrary.linear_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=2, num_item_features=2).build(),
     "LinearKnapsack", np.array([1.0, 1.0, 1.0, 1.0]), False, None),
    ("3.PlainSingleItem", ScenarioLibrary.plain_single_item().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=4, num_item_features=1).build(),
     "PlainSingleItem", np.array([1.0, 1.0, 1.0, 1.0, 1.0]), False, None),
    ("4.QuadKnapsack", ScenarioLibrary.quadratic_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build(),
     "QuadKnapsack", np.array([1.0, 1.0, 1.0, 1.0]), False, None),
    ("5.QuadSuper", ScenarioLibrary.quadratic_supermodular().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build(),
     "QuadSupermodularNetwork", np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]), False, None),
]

for name, scenario, subproblem, theta, use_greedy, theta_lbs in scenarios:
    r = benchmark(name, scenario, subproblem, theta, use_greedy, theta_lbs)
    if r:
        results.append(r)
    comm.Barrier()

# Summary
if rank == 0 and results:
    avg = np.mean([r['speedup'] for r in results])
    print(f"\n{'='*70}")
    print(f"Average speedup: {avg:.2f}x")
    print("="*70)
