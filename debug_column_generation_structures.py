#!/usr/bin/env python
"""
Debug column generation on richer problem structures (Greedy, Knapsack).
Verifies row-generation and column-generation deliver matching objectives.
Always wraps execution in a 15-second timeout.
"""

import signal
from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import numpy as np
from mpi4py import MPI

from bundlechoice import BundleChoice


@contextmanager
def timeout(seconds: int):
    """Abort after `seconds` (POSIX only)."""

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def base_config(num_agents: int, num_items: int, num_features: int, subproblem_name: str, num_simuls: int = 1) -> Dict:
    return {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {"name": subproblem_name},
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 1e-5,
            "tol_row_generation": 0.0,
            "row_generation_decay": 0.0,
            "gurobi_settings": {"OutputFlag": 0},
        },
    }


def make_greedy_case(seed: int = 10) -> Tuple[Dict, Dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_agents, num_items, num_features = 3, 5, 4
    modular = rng.normal(size=(num_agents, num_items, num_features))
    errors = rng.normal(scale=0.2, size=(1, num_agents, num_items))
    config = base_config(num_agents, num_items, num_features, "Greedy", num_simuls=1)
    input_data = {"agent_data": {"modular": modular}, "errors": errors}
    theta_true = rng.normal(size=num_features)
    return config, input_data, theta_true


def make_linear_knapsack_case(seed: int = 20) -> Tuple[Dict, Dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_agents, num_items = 3, 6
    mod_agent, mod_item = 2, 2
    num_features = mod_agent + mod_item

    modular_agent = rng.normal(size=(num_agents, num_items, mod_agent))
    modular_item = rng.normal(size=(num_items, mod_item))
    weights = rng.integers(1, 6, size=num_items)
    capacity = rng.integers(int(0.4 * weights.sum()), int(0.6 * weights.sum()) + 1, size=num_agents)
    errors = rng.normal(scale=0.25, size=(1, num_agents, num_items))

    config = base_config(num_agents, num_items, num_features, "LinearKnapsack", num_simuls=1)
    input_data = {
        "agent_data": {"modular": modular_agent, "capacity": capacity},
        "item_data": {"modular": modular_item, "weights": weights},
        "errors": errors,
    }
    theta_true = rng.normal(size=num_features)
    return config, input_data, theta_true


def make_quad_knapsack_case(seed: int = 30) -> Tuple[Dict, Dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_agents, num_items = 3, 5
    mod_agent, quad_agent = 2, 1
    mod_item, quad_item = 1, 1
    num_features = mod_agent + quad_agent + mod_item + quad_item

    modular_agent = rng.normal(scale=0.5, size=(num_agents, num_items, mod_agent))
    modular_item = rng.normal(scale=0.5, size=(num_items, mod_item))

    agent_quad = rng.normal(scale=0.2, size=(num_agents, num_items, num_items, quad_agent))
    item_quad = rng.normal(scale=0.2, size=(num_items, num_items, quad_item))
    for a in range(num_agents):
        for f in range(quad_agent):
            mat = agent_quad[a, :, :, f]
            mat = 0.5 * (mat + mat.T)
            np.fill_diagonal(mat, 0.0)
            agent_quad[a, :, :, f] = mat
    for f in range(quad_item):
        mat = item_quad[:, :, f]
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        item_quad[:, :, f] = mat

    weights = rng.integers(1, 5, size=num_items)
    capacity = rng.integers(int(0.45 * weights.sum()), int(0.65 * weights.sum()) + 1, size=num_agents)
    errors = rng.normal(scale=0.3, size=(1, num_agents, num_items))

    config = base_config(num_agents, num_items, num_features, "QuadKnapsack", num_simuls=1)
    input_data = {
        "agent_data": {"modular": modular_agent, "quadratic": agent_quad, "capacity": capacity},
        "item_data": {"modular": modular_item, "quadratic": item_quad, "weights": weights},
        "errors": errors,
    }
    theta_true = rng.normal(size=num_features)
    return config, input_data, theta_true


def make_quad_supermod_case(seed: int = 40) -> Tuple[Dict, Dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_agents, num_items = 3, 5
    mod_agent, quad_agent = 2, 1
    mod_item, quad_item = 1, 1
    num_features = mod_agent + quad_agent + mod_item + quad_item

    modular_agent = rng.normal(scale=0.4, size=(num_agents, num_items, mod_agent))
    modular_item = rng.normal(scale=0.4, size=(num_items, mod_item))

    agent_quad = rng.uniform(0.0, 0.5, size=(num_agents, num_items, num_items, quad_agent))
    item_quad = rng.uniform(0.0, 0.5, size=(num_items, num_items, quad_item))
    for a in range(num_agents):
        for f in range(quad_agent):
            mat = agent_quad[a, :, :, f]
            mat = -np.triu(np.abs(mat), k=1)
            agent_quad[a, :, :, f] = mat
    for f in range(quad_item):
        mat = item_quad[:, :, f]
        mat = -np.triu(np.abs(mat), k=1)
        item_quad[:, :, f] = mat

    errors = rng.normal(scale=0.25, size=(1, num_agents, num_items))

    config = base_config(num_agents, num_items, num_features, "QuadSupermodularNetwork", num_simuls=1)
    input_data = {
        "agent_data": {"modular": modular_agent, "quadratic": agent_quad},
        "item_data": {"modular": modular_item, "quadratic": item_quad},
        "errors": errors,
    }
    theta_true = np.abs(rng.normal(size=num_features))
    return config, input_data, theta_true


CASE_BUILDERS: Dict[str, Callable[[], Tuple[Dict, Dict, np.ndarray]]] = {
    "Greedy": make_greedy_case,
    "LinearKnapsack": make_linear_knapsack_case,
}


def run_case(name: str, builder: Callable[[], Tuple[Dict, Dict, np.ndarray]]):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config, input_data, theta_true = builder()
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(input_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()

    obs_bundles = bc.subproblems.init_and_solve(theta_true)
    if rank == 0:
        expected_shape = (bc.config.dimensions.num_agents, bc.config.dimensions.num_items)
        if obs_bundles.shape != expected_shape:
            try:
                reshaped = obs_bundles.reshape(
                    bc.config.dimensions.num_simuls,
                    bc.config.dimensions.num_agents,
                    bc.config.dimensions.num_items,
                )
                obs_bundles = reshaped[0]
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected obs_bundles shape {obs_bundles.shape}, expected {expected_shape}"
                ) from exc
        bc.data.input_data["obs_bundle"] = obs_bundles
    bc.data.load_and_scatter(bc.data.input_data if rank == 0 else None)
    bc.features.build_from_data()
    bc.subproblems.load()

    if rank == 0:
        print(f"\n=== {name} ===")

    theta_row = bc.row_generation.solve()
    obj_row = bc.row_generation.objective(theta_row) if bc.is_root() else None

    theta_col = bc.column_generation.solve()
    obj_col = bc.column_generation.objective(theta_col) if bc.is_root() else None

    if rank == 0:
        diff = abs(obj_row - obj_col)
        print(f"Row theta:         {theta_row}")
        print(f"Column theta:      {theta_col}")
        print(f"Row objective:     {obj_row:.6f}")
        print(f"Column objective:  {obj_col:.6f}")
        print(f"Objective diff:    {diff:.3e}")
        assert diff <= 1e-4, f"Objective mismatch in {name}: {diff:.3e}"
        print(f"✓ {name} objectives match")


def main():
    with timeout(15):
        for name, builder in CASE_BUILDERS.items():
            run_case(name, builder)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n✓ All extended column-generation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n✗ {exc}")
        raise SystemExit(124)
    except Exception as exc:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n✗ Extended debug failed: {exc}")
        raise

