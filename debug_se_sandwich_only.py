#!/usr/bin/env python3
"""Compute sandwich SE for all 4 scenarios and output clean tables."""

import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

NUM_AGENTS = 150
NUM_ITEMS = 15
NUM_SIMULS_SANDWICH = 100
SEED = 42


def greedy_oracle(agent_idx: int, bundles: np.ndarray, data: dict) -> np.ndarray:
    modular = data["agent_data"]["modular"][agent_idx]
    modular = np.atleast_2d(modular)
    single = bundles.ndim == 1
    if single:
        bundles = bundles[:, None]
    modular_feat = modular.T @ bundles
    quad_feat = -np.sum(bundles, axis=0, keepdims=True) ** 2
    features = np.vstack((modular_feat, quad_feat))
    return features[:, 0] if single else features


def run_scenario(name, scenario, bc_est, theta_true):
    """Run one scenario with sandwich SE only."""
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
    
    result = bc_est.row_generation.solve()
    theta_hat = result.theta_hat if rank == 0 else None
    theta_hat = comm.bcast(theta_hat, root=0)
    
    se_result = bc_est.standard_errors.compute(theta_hat, num_simulations=NUM_SIMULS_SANDWICH, seed=1995)
    
    if rank == 0 and se_result is not None:
        se = se_result.se
        print(f"\n{'Param':<8} {'True':<8} {'Est':<10} {'SE':<10} {'t-stat':<10} {'|Err|/SE':<10}")
        print("-" * 60)
        for i in range(len(theta_true)):
            true_val = theta_true[i]
            est_val = theta_hat[i]
            err = est_val - true_val
            t_stat = est_val / se[i] if se[i] > 0 else np.nan
            err_se = abs(err) / se[i] if se[i] > 0 else np.nan
            print(f"θ[{i}]    {true_val:<8.3f} {est_val:<10.4f} {se[i]:<10.4f} {t_stat:<10.2f} {err_se:<10.2f}")
    
    return theta_hat, se_result.se if rank == 0 and se_result else None


# Greedy
if rank == 0:
    print("\n" + "=" * 70)
    print("Running: Greedy")
    print("=" * 70)

theta_true = np.array([1.0, 1.0, 0.1])
NUM_FEATURES = 3
scenario = ScenarioLibrary.greedy().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_num_features(NUM_FEATURES).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": NUM_FEATURES, "num_simulations": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS_SANDWICH, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.set_features_oracle(greedy_oracle)
bc.subproblems.load()
from bundlechoice.subproblems.registry.greedy import GreedySubproblem
from bundlechoice.scenarios.greedy import _install_find_best_item
if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
    _install_find_best_item(bc.subproblems.subproblem_instance)
bc.subproblems.initialize_local()

greedy_result = run_scenario("Greedy", scenario, bc, theta_true)
comm.Barrier()


# LinearKnapsack
if rank == 0:
    print("\n" + "=" * 70)
    print("Running: LinearKnapsack")
    print("=" * 70)

theta_true = np.array([1.0, 1.0, 1.0, 1.0])
scenario = ScenarioLibrary.linear_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=2, num_item_features=2).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 4, "num_simulations": 1},
    "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 0.5}},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS_SANDWICH, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

knapsack_result = run_scenario("LinearKnapsack", scenario, bc, theta_true)
comm.Barrier()


# PlainSingleItem
if rank == 0:
    print("\n" + "=" * 70)
    print("Running: PlainSingleItem")
    print("=" * 70)

theta_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
scenario = ScenarioLibrary.plain_single_item().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=4, num_item_features=1).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 5, "num_simulations": 1},
    "subproblem": {"name": "PlainSingleItem"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS_SANDWICH, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

plain_result = run_scenario("PlainSingleItem", scenario, bc, theta_true)
comm.Barrier()


# QuadSupermodular
if rank == 0:
    print("\n" + "=" * 70)
    print("Running: QuadSupermodular")
    print("=" * 70)

theta_true = np.array([0.4, 0.3, 0.4, 0.3, 0.2, -0.05])
scenario = ScenarioLibrary.quadratic_supermodular().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 6, "num_simulations": 1},
    "subproblem": {"name": "QuadSupermodularNetwork"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS_SANDWICH, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

quad_result = run_scenario("QuadSupermodular", scenario, bc, theta_true)

if rank == 0:
    print("\n" + "=" * 70)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 70)
