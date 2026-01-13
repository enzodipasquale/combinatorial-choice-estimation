#!/usr/bin/env python3
"""SE comparison - All three methods on FIVE settings."""

import numpy as np
from mpi4py import MPI
import sys, os, logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank != 0:
    sys.stdout = open(os.devnull, 'w')
logging.disable(logging.CRITICAL)

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

NUM_AGENTS = 150
NUM_ITEMS = 15
NUM_SE_SIMULS = 130
NUM_BOOT = 100
SEED = 42


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


def run(name, theta_true, scenario, subproblem, use_greedy=False, prepared=None, theta_lbs=None):
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    if prepared is None:
        prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)
    
    bc = BundleChoice()
    config = {
        "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": len(theta_true), "num_simulations": 1},
        "subproblem": {"name": subproblem, "settings": {"TimeLimit": 0.5} if "Knapsack" in subproblem else {}},
        "row_generation": {"max_iters": 200, "theta_ubs": 100},
    }
    if theta_lbs is not None:
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
    
    result = bc.row_generation.solve()
    theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
    
    # 1. Sandwich SE
    se_sand = None
    try:
        se_result = bc.standard_errors.compute(theta_hat, num_simulations=NUM_SE_SIMULS, seed=1995)
        if se_result:
            se_sand = se_result.se
    except Exception as e:
        if rank == 0:
            print(f"Sandwich failed: {e}")
    
    # 2. Bootstrap SE
    se_boot = None
    def bootstrap_solve(boot_data):
        bc.data.load_and_scatter(boot_data)
        if use_greedy:
            bc.oracles.set_features_oracle(greedy_oracle)
        else:
            bc.oracles.build_from_data()
        bc.subproblems.initialize_local()
        result = bc.row_generation.solve()
        return result.theta_hat if bc.is_root() else None
    
    try:
        se_result = bc.standard_errors.compute_bootstrap(
            theta_hat=theta_hat,
            solve_fn=bootstrap_solve,
            num_bootstrap=NUM_BOOT,
            seed=SEED,
        )
        if se_result:
            se_boot = se_result.se
    except Exception as e:
        if rank == 0:
            print(f"Bootstrap failed: {e}")
    
    # 3. Bayesian Bootstrap SE - must restore state after Bootstrap corrupted it
    se_bayes = None
    try:
        # Restore original data (Bootstrap corrupted bc.data with resampled data)
        bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
        if use_greedy:
            bc.oracles.set_features_oracle(greedy_oracle)
        else:
            bc.oracles.build_from_data()
        bc.subproblems.initialize_local()
        # Re-solve to rebuild master model with original data
        bc.row_generation.solve()
        
        se_result = bc.standard_errors.compute_bayesian_bootstrap(
            theta_hat=theta_hat,
            row_generation=bc.row_generation,
            num_bootstrap=NUM_BOOT,
            seed=SEED,
        )
        if se_result:
            se_bayes = se_result.se
    except Exception as e:
        if rank == 0:
            print(f"Bayesian failed: {e}")
    
    print(f"\n{'Param':<6} {'True':<8} {'Est':<10} {'SE(Sand)':<10} {'SE(Boot)':<10} {'SE(Bayes)':<10}")
    print("-" * 65)
    for i in range(len(theta_true)):
        t, e = theta_true[i], theta_hat[i]
        s1 = se_sand[i] if se_sand is not None else np.nan
        s2 = se_boot[i] if se_boot is not None else np.nan
        s3 = se_bayes[i] if se_bayes is not None else np.nan
        print(f"θ[{i}]  {t:<8.3f} {e:<10.4f} {s1:<10.4f} {s2:<10.4f} {s3:<10.4f}")


print(f"SE COMPARISON: N={NUM_AGENTS}, J={NUM_ITEMS}, Simuls={NUM_SE_SIMULS}, Boot={NUM_BOOT}")

# 1. Greedy
scenario = ScenarioLibrary.greedy().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_num_features(3).build()
run("Greedy", np.array([1.0, 1.0, 0.1]), scenario, "Greedy", use_greedy=True)
comm.Barrier()

# 2. LinearKnapsack
scenario = ScenarioLibrary.linear_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=2, num_item_features=2).build()
run("LinearKnapsack", np.array([1.0, 1.0, 1.0, 1.0]), scenario, "LinearKnapsack")
comm.Barrier()

# 3. PlainSingleItem
scenario = ScenarioLibrary.plain_single_item().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=4, num_item_features=1).build()
run("PlainSingleItem", np.array([1.0, 1.0, 1.0, 1.0, 1.0]), scenario, "PlainSingleItem")
comm.Barrier()

# 4. QuadraticKnapsack (default: 1 agent_mod + 1 agent_quad + 1 item_mod + 1 item_quad = 4 features)
scenario = ScenarioLibrary.quadratic_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build()
run("QuadraticKnapsack", np.array([1.0, 1.0, 1.0, 1.0]), scenario, "QuadKnapsack")
comm.Barrier()

# 5. QuadSupermodular - needs negative bounds for identification!
theta_quad = np.array([-0.1, -0.1, -0.1, -0.1, -0.5, -0.5])
scenario = ScenarioLibrary.quadratic_supermodular().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build()
prepared_quad = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED, theta=theta_quad)
if rank == 0:
    sizes = prepared_quad.estimation_data["obs_bundle"].sum(axis=1)
    print(f"QuadSupermodular bundle sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
run("QuadSupermodular", theta_quad, scenario, "QuadSupermodularNetwork", prepared=prepared_quad, 
    theta_lbs=[-100]*6)  # Allow negative theta!

print("\n" + "="*80 + "\nDONE\n" + "="*80)
