#!/usr/bin/env python
"""
Regression checks for column generation vs row generation under a timeout wrapper.
Exercises multiple subproblem structures (greedy, knapsacks, supermodular).
"""

from __future__ import annotations

import copy
import signal
from contextlib import contextmanager
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI

from bundlechoice import BundleChoice


@contextmanager
def timeout(seconds: int):
    """Abort block if it exceeds `seconds` (POSIX only)."""

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def compare_row_column(
    name: str,
    config: Dict,
    input_data: Dict | None,
    theta_seed: int,
    objective_tol: float = 1e-4,
) -> None:
    """Run row and column generation on a dataset and compare objectives."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()
    bc.subproblems.load()

    rng = np.random.default_rng(theta_seed)
    theta_true = rng.uniform(-0.5, 0.5, size=config["dimensions"]["num_features"])
    obs_bundles = bc.subproblems.init_and_solve(theta_true)

    if rank == 0:
        input_with_obs = copy.deepcopy(input_data) if input_data is not None else {}
        input_with_obs["obs_bundle"] = obs_bundles
    else:
        input_with_obs = None

    bc.data.load_and_scatter(input_with_obs)
    bc.features.build_from_data()
    bc.subproblems.load()

    theta_row = bc.row_generation.solve()
    row_obj = getattr(bc.row_generation.master_model, "ObjVal", None)

    theta_col = bc.column_generation.solve()
    col_obj = getattr(bc.column_generation.master_model, "ObjVal", None)

    if rank == 0:
        print(f"\n=== {name} ===")
        print("  row objective    =", row_obj)
        print("  column objective =", col_obj)
        if row_obj is not None and col_obj is not None:
            diff = abs(row_obj - col_obj)
            print("  objective diff   =", diff)
            if diff > objective_tol:
                col_solver = bc.column_generation
                manual = 0.0
                for var, column in zip(col_solver.column_vars, col_solver.active_columns):
                    manual += float(var.X) * float(column["errors"])
                if col_solver.alpha_vars:
                    for idx, alpha in enumerate(col_solver.alpha_vars):
                        if alpha is not None:
                            manual -= float(col_solver.theta_upper[idx]) * float(alpha.X)
                if col_solver.beta_vars:
                    for idx, beta in enumerate(col_solver.beta_vars):
                        if beta is not None:
                            manual += float(col_solver.theta_lower[idx]) * float(beta.X)
                print("  manual dual objective =", manual)
                assert diff <= objective_tol, f"Objective mismatch ({diff:.3e})"
        theta_gap = float(np.linalg.norm(theta_row - theta_col))
        print("  theta diff norm  =", theta_gap)
        print("✓ objectives match")


# --------------------------------------------------------------------------- #
# Data generators for different subproblem structures
# --------------------------------------------------------------------------- #


def build_greedy_case() -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(101)
    num_agents, num_items, num_features = 6, 9, 4
    num_simuls = 1

    agent_modular = rng.normal(scale=0.7, size=(num_agents, num_items, num_features))
    errors = rng.normal(scale=0.2, size=(num_simuls, num_agents, num_items))

    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": "Greedy"},
        "row_generation": {
            "max_iters": 60,
            "tolerance_optimality": 1e-5,
            "theta_ubs": 5.0,
            "theta_lbs": -5.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }

    input_data = {
        "agent_data": {"modular": agent_modular},
        "errors": errors,
    }
    return config, input_data


def build_plain_single_item_case() -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(202)
    num_agents, num_items, num_features = 6, 8, 3
    num_simuls = 1

    agent_modular = rng.normal(scale=0.5, size=(num_agents, num_items, num_features))
    errors = rng.normal(scale=0.1, size=(num_simuls, num_agents, num_items))

    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": "PlainSingleItem"},
        "row_generation": {
            "max_iters": 40,
            "tolerance_optimality": 1e-5,
            "theta_ubs": 5.0,
            "theta_lbs": -5.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }

    input_data = {
        "agent_data": {"modular": agent_modular},
        "errors": errors,
    }
    return config, input_data


def build_linear_knapsack_case() -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(303)
    num_agents, num_items = 5, 10
    num_simuls = 1
    agent_mod_dim, item_mod_dim = 2, 2
    num_features = agent_mod_dim + item_mod_dim

    agent_modular = rng.normal(scale=0.6, size=(num_agents, num_items, agent_mod_dim))
    item_modular = rng.normal(scale=0.4, size=(num_items, item_mod_dim))
    weights = rng.uniform(1.0, 4.0, size=num_items)
    capacity = rng.uniform(weights.sum() * 0.3, weights.sum() * 0.6, size=num_agents)
    errors = rng.normal(scale=0.15, size=(num_simuls, num_agents, num_items))

    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": "LinearKnapsack"},
        "row_generation": {
            "max_iters": 80,
            "tolerance_optimality": 1e-5,
            "theta_ubs": 5.0,
            "theta_lbs": -5.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }

    input_data = {
        "agent_data": {"modular": agent_modular, "capacity": capacity},
        "item_data": {"modular": item_modular, "weights": weights},
        "errors": errors,
    }
    return config, input_data


def build_quadratic_knapsack_case() -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(404)
    num_agents, num_items = 4, 7
    num_simuls = 1
    agent_mod_dim, item_mod_dim = 1, 1
    agent_quad_dim, item_quad_dim = 1, 1
    num_features = agent_mod_dim + agent_quad_dim + item_mod_dim + item_quad_dim

    agent_modular = rng.normal(scale=0.5, size=(num_agents, num_items, agent_mod_dim))

    agent_quadratic = rng.uniform(0.0, 0.3, size=(num_agents, num_items, num_items, agent_quad_dim))
    for d in range(agent_quad_dim):
        for i in range(num_items):
            agent_quadratic[:, i, i, d] = 0.0
        agent_quadratic[..., d] = np.triu(agent_quadratic[..., d], k=1)

    item_modular = rng.normal(scale=0.4, size=(num_items, item_mod_dim))

    item_quadratic = rng.uniform(0.0, 0.25, size=(num_items, num_items, item_quad_dim))
    for d in range(item_quad_dim):
        for i in range(num_items):
            item_quadratic[i, i, d] = 0.0
        item_quadratic[..., d] = np.triu(item_quadratic[..., d], k=1)

    weights = rng.uniform(1.0, 3.0, size=num_items)
    capacity = rng.uniform(weights.sum() * 0.35, weights.sum() * 0.65, size=num_agents)
    errors = rng.normal(scale=0.1, size=(num_simuls, num_agents, num_items))

    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": "QuadKnapsack"},
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 1e-5,
            "theta_ubs": 5.0,
            "theta_lbs": -5.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }

    input_data = {
        "agent_data": {
            "modular": agent_modular,
            "quadratic": agent_quadratic,
            "capacity": capacity,
        },
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic,
            "weights": weights,
        },
        "errors": errors,
    }
    return config, input_data


def build_quad_supermod_case() -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(505)
    num_agents, num_items = 3, 6
    num_simuls = 1
    agent_mod_dim, item_mod_dim = 2, 1
    agent_quad_dim, item_quad_dim = 1, 2
    num_features = agent_mod_dim + agent_quad_dim + item_mod_dim + item_quad_dim

    agent_modular = rng.normal(scale=0.5, size=(num_agents, num_items, agent_mod_dim))

    agent_quadratic = rng.uniform(0.0, 0.2, size=(num_agents, num_items, num_items, agent_quad_dim))
    for d in range(agent_quad_dim):
        for i in range(num_items):
            agent_quadratic[:, i, i, d] = 0.0
        agent_quadratic[..., d] = -np.triu(np.abs(agent_quadratic[..., d]), k=1)

    item_modular = rng.normal(scale=0.4, size=(num_items, item_mod_dim))

    item_quadratic = rng.uniform(0.0, 0.15, size=(num_items, num_items, item_quad_dim))
    for d in range(item_quad_dim):
        for i in range(num_items):
            item_quadratic[i, i, d] = 0.0
        item_quadratic[..., d] = -np.triu(np.abs(item_quadratic[..., d]), k=1)

    errors = rng.normal(scale=0.1, size=(num_simuls, num_agents, num_items))

    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": "QuadSupermodularNetwork"},
        "row_generation": {
            "max_iters": 120,
            "tolerance_optimality": 5e-5,
            "theta_ubs": 5.0,
            "theta_lbs": -5.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }

    input_data = {
        "agent_data": {
            "modular": agent_modular,
            "quadratic": agent_quadratic,
        },
        "item_data": {
            "modular": item_modular,
            "quadratic": item_quadratic,
        },
        "errors": errors,
    }
    return config, input_data


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cases = [
        ("Greedy (medium)", build_greedy_case, 0),
        ("Plain Single Item", build_plain_single_item_case, 1),
        ("Linear Knapsack", build_linear_knapsack_case, 2),
    ]

    try:
        with timeout(15):
            for name, builder, seed in cases:
                if rank == 0:
                    cfg, data = builder()
                else:
                    cfg, data = None, None
                cfg = comm.bcast(cfg, root=0)
                compare_row_column(name, cfg, data if rank == 0 else None, theta_seed=seed)
            if rank == 0:
                print("\n✓ All column generation tests passed.")
    except TimeoutError as exc:
        if rank == 0:
            print(f"\n✗ {exc}")
        raise SystemExit(1)
    except Exception as exc:
        if rank == 0:
            print(f"\n✗ Column generation tests failed: {exc}")
        raise
