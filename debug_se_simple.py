#!/usr/bin/env python3
"""Simple SE test for all 4 scenarios."""

import sys
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Force flush
def log(msg):
    if rank == 0:
        print(msg, flush=True)
        sys.stdout.flush()

NUM_AGENTS = 150
NUM_ITEMS = 15
NUM_SIMULS = 100
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


def print_table(name, theta_true, theta_hat, se):
    """Print result table."""
    log(f"\n{'='*70}")
    log(f"{name}")
    log(f"{'='*70}")
    log(f"{'Param':<8} {'True':<8} {'Est':<10} {'SE':<10} {'t-stat':<10} {'|Err|/SE':<10}")
    log("-" * 60)
    for i in range(len(theta_true)):
        t = theta_true[i]
        e = theta_hat[i]
        s = se[i]
        err = e - t
        t_stat = e / s if s > 0 else np.nan
        err_se = abs(err) / s if s > 0 else np.nan
        log(f"θ[{i}]    {t:<8.3f} {e:<10.4f} {s:<10.4f} {t_stat:<10.2f} {err_se:<10.2f}")


# =====================================================================
# GREEDY
# =====================================================================
log("\n>>> GREEDY")

theta_true = np.array([1.0, 1.0, 0.1])
NUM_FEATURES = 3
scenario = ScenarioLibrary.greedy().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_num_features(NUM_FEATURES).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": NUM_FEATURES, "num_simulations": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.set_features_oracle(greedy_oracle)
bc.subproblems.load()
from bundlechoice.subproblems.registry.greedy import GreedySubproblem
from bundlechoice.scenarios.greedy import _install_find_best_item
if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
    _install_find_best_item(bc.subproblems.subproblem_instance)
bc.subproblems.initialize_local()

result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
se_result = bc.standard_errors.compute(theta_hat, num_simulations=NUM_SIMULS, seed=1995)

if rank == 0 and se_result:
    print_table("GREEDY", theta_true, theta_hat, se_result.se)

comm.Barrier()


# =====================================================================
# LINEAR KNAPSACK
# =====================================================================
log("\n>>> LINEAR KNAPSACK")

theta_true = np.array([1.0, 1.0, 1.0, 1.0])
scenario = ScenarioLibrary.linear_knapsack().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=2, num_item_features=2).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 4, "num_simulations": 1},
    "subproblem": {"name": "LinearKnapsack", "settings": {"TimeLimit": 0.5}},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
se_result = bc.standard_errors.compute(theta_hat, num_simulations=NUM_SIMULS, seed=1995)

if rank == 0 and se_result:
    print_table("LINEAR KNAPSACK", theta_true, theta_hat, se_result.se)

comm.Barrier()


# =====================================================================
# PLAIN SINGLE ITEM
# =====================================================================
log("\n>>> PLAIN SINGLE ITEM")

theta_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
scenario = ScenarioLibrary.plain_single_item().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).with_feature_counts(num_agent_features=4, num_item_features=1).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 5, "num_simulations": 1},
    "subproblem": {"name": "PlainSingleItem"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
se_result = bc.standard_errors.compute(theta_hat, num_simulations=NUM_SIMULS, seed=1995)

if rank == 0 and se_result:
    print_table("PLAIN SINGLE ITEM", theta_true, theta_hat, se_result.se)

comm.Barrier()


# =====================================================================
# QUAD SUPERMODULAR
# =====================================================================
log("\n>>> QUAD SUPERMODULAR")

theta_true = np.array([0.4, 0.3, 0.4, 0.3, 0.2, -0.05])
scenario = ScenarioLibrary.quadratic_supermodular().with_dimensions(num_agents=NUM_AGENTS, num_items=NUM_ITEMS).build()
prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=SEED)

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": NUM_AGENTS, "num_items": NUM_ITEMS, "num_features": 6, "num_simulations": 1},
    "subproblem": {"name": "QuadSupermodularNetwork"},
    "row_generation": {"max_iters": 200, "theta_ubs": 100},
    "standard_errors": {"num_simulations": NUM_SIMULS, "step_size": 1e-2},
})
bc.data.load_and_scatter(prepared.estimation_data if rank == 0 else None)
bc.oracles.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
se_result = bc.standard_errors.compute(theta_hat, num_simulations=NUM_SIMULS, seed=1995)

if rank == 0 and se_result:
    print_table("QUAD SUPERMODULAR", theta_true, theta_hat, se_result.se)


log("\n" + "=" * 70)
log("ALL DONE")
log("=" * 70)
