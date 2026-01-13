#!/usr/bin/env python3
"""
Benchmark row vs column generation on the down-sized data-generating
processes from `benchmarking/greedy`, `benchmarking/plain_single_item`,
and `benchmarking/knapsack` (300 agents, 100 items, few features).

For each scenario we:
  1. Recreate the synthetic data exactly as in the benchmark folder.
  2. Solve with row generation and column generation.
  3. Record runtimes, objective values, and parameter discrepancies.

Results are appended to `benchmarking/benchmark_row_vs_column_results.csv`
on rank 0.  Launch this script with `mpirun`.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from mpi4py import MPI

from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.scenarios.data_generator import QuadraticGenerationMethod

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
IS_ROOT = RANK == 0

RESULTS_PATH = Path(__file__).with_name("benchmark_row_vs_column_results.csv")


@contextmanager
def maybe_timeout(seconds: Optional[int]):
    """POSIX alarm-based timeout (disabled when seconds is None or <=0)."""

    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class BenchmarkResult:
    scenario: str
    status: str
    num_agents: int
    num_items: int
    num_features: int
    num_simulations: int
    sigma: Optional[float]
    row_time: Optional[float]
    column_time: Optional[float]
    row_objective: Optional[float]
    column_objective: Optional[float]
    objective_gap: Optional[float]
    theta_norm_diff: Optional[float]
    timestamp: str

    def to_row(self) -> Dict[str, Optional[float]]:
        return {
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "status": self.status,
            "num_agents": self.num_agents,
            "num_items": self.num_items,
            "num_features": self.num_features,
            "num_simulations": self.num_simulations,
            "sigma": self.sigma,
            "row_time": self.row_time,
            "column_time": self.column_time,
            "row_objective": self.row_objective,
            "column_objective": self.column_objective,
            "objective_gap": self.objective_gap,
            "theta_norm_diff": self.theta_norm_diff,
        }


def barrier():
    """Convenience MPI barrier."""
    COMM.Barrier()


def timed_solver(callable_fn: Callable[[], np.ndarray],
                 timeout: Optional[int],
                 label: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Run `callable_fn` across all ranks, measuring elapsed time (root only).
    Returns (result, elapsed_seconds).  On timeout, returns (None, None).
    """
    barrier()
    start = time.perf_counter()
    try:
        with maybe_timeout(timeout):
            result = callable_fn()
        barrier()
        elapsed = time.perf_counter() - start
    except TimeoutError:
        if IS_ROOT:
            print(f"[{label}] timed out after {timeout} seconds", flush=True)
        barrier()
        return None, None
    return result, elapsed


def broadcast_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Broadcast numpy array from root to all ranks."""
    array = COMM.bcast(array if IS_ROOT else None, root=0)
    if array is None:
        return None
    return np.asarray(array)


def run_methods(name: str,
                bc: BundleChoice,
                theta_star: np.ndarray,
                metadata: Dict[str, Optional[float]],
                timeout: Optional[int]) -> BenchmarkResult:
    """
    Run row and column generation on the prepared `BundleChoice` instance.
    """
    sigma = metadata.get("sigma")
    num_agents = metadata["num_agents"]
    num_items = metadata["num_items"]
    num_features = metadata["num_features"]
    num_simulations = metadata["num_simulations"]

    # Ensure subproblems ready
    bc.subproblems.load()

    # Row generation
    result_row, row_time = timed_solver(bc.row_generation.solve, timeout, f"{name}-row")
    if result_row is not None:
        theta_row = result_row.theta_hat if hasattr(result_row, 'theta_hat') else result_row
    else:
        theta_row = None
    theta_row = broadcast_array(theta_row)

    if theta_row is None:
        return BenchmarkResult(
            scenario=name,
            status="row_timeout",
            num_agents=num_agents,
            num_items=num_items,
            num_features=num_features,
            num_simulations=num_simulations,
            sigma=sigma,
            row_time=None,
            column_time=None,
            row_objective=None,
            column_objective=None,
            objective_gap=None,
            theta_norm_diff=None,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    # Column generation
    result_col, column_time = timed_solver(bc.column_generation.solve, timeout, f"{name}-column")
    if result_col is not None:
        theta_col = result_col.theta_hat if hasattr(result_col, 'theta_hat') else result_col
    else:
        theta_col = None
    theta_col = broadcast_array(theta_col)

    if theta_col is None:
        return BenchmarkResult(
            scenario=name,
            status="column_timeout",
            num_agents=num_agents,
            num_items=num_items,
            num_features=num_features,
            num_simulations=num_simulations,
            sigma=sigma,
            row_time=row_time if IS_ROOT else None,
            column_time=None,
            row_objective=None,
            column_objective=None,
            objective_gap=None,
            theta_norm_diff=None,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    # Evaluate objectives (all ranks participate)
    row_obj = bc.row_generation.objective(theta_row)
    column_obj = bc.column_generation.objective(theta_col)

    if IS_ROOT:
        objective_gap = abs(row_obj - column_obj) if row_obj is not None and column_obj is not None else None
        theta_norm_diff = np.linalg.norm(theta_row - theta_col)
        status = "ok"
        timestamp = datetime.now().isoformat(timespec="seconds")
        result = BenchmarkResult(
            scenario=name,
            status=status,
            num_agents=num_agents,
            num_items=num_items,
            num_features=num_features,
            num_simulations=num_simulations,
            sigma=sigma,
            row_time=row_time,
            column_time=column_time,
            row_objective=row_obj,
            column_objective=column_obj,
            objective_gap=objective_gap,
            theta_norm_diff=theta_norm_diff,
            timestamp=timestamp,
        )
    else:
        result = BenchmarkResult(
            scenario=name,
            status="worker",
            num_agents=num_agents,
            num_items=num_items,
            num_features=num_features,
            num_simulations=num_simulations,
            sigma=sigma,
            row_time=None,
            column_time=None,
            row_objective=None,
            column_objective=None,
            objective_gap=None,
            theta_norm_diff=None,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    barrier()
    return result


def setup_greedy(timeout: Optional[int], seed: Optional[int] = None) -> BenchmarkResult:
    num_agents = 300
    num_items = 100
    num_features = 4
    num_simulations = 1
    sigma = 1.0

    # Use factory to generate data (matches manual generation)
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_num_features(num_features)
        .with_num_simulations(num_simulations)
        .with_sigma(sigma)
        .build()
    )

    # Use default theta_star (all ones)
    theta_star = np.ones(num_features)
    
    # Prepare with theta_star to generate bundles
    prepared = scenario.prepare(comm=COMM, timeout_seconds=timeout, seed=seed, theta=theta_star)

    bc = BundleChoice()
    prepared.apply(bc, comm=COMM, stage="estimation")

    metadata = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "sigma": sigma,
    }

    return run_methods("greedy", bc, theta_star, metadata, timeout)


def setup_plain_single_item(timeout: Optional[int], seed: Optional[int] = None) -> BenchmarkResult:
    num_agents = 300
    num_items = 100
    num_features = 4
    num_simulations = 1
    sigma = 1.0

    # Use factory to generate data (matches manual generation with correlation)
    scenario = (
        ScenarioLibrary.plain_single_item()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(num_agent_features=num_features, num_item_features=0)
        .with_num_simulations(num_simulations)
        .with_sigma(sigma)
        .with_correlation(enabled=True, matrix_range=(0, 4), normalize=True)  # Match manual
        .build()
    )

    # Use default theta_star (all ones)
    theta_star = np.ones(num_features)
    
    # Prepare with theta_star to generate bundles
    prepared = scenario.prepare(comm=COMM, timeout_seconds=timeout, seed=seed, theta=theta_star)

    bc = BundleChoice()
    prepared.apply(bc, comm=COMM, stage="estimation")

    metadata = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "sigma": sigma,
    }

    return run_methods("plain_single_item", bc, theta_star, metadata, timeout)


def setup_knapsack(timeout: Optional[int], seed: Optional[int] = None) -> BenchmarkResult:
    num_agents = 300
    num_items = 100
    num_simulations = 1
    modular_agent_features = 2
    modular_item_features = 1
    num_features = modular_agent_features + modular_item_features
    sigma = 1.0

    # Use factory to generate data (matches manual generation)
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(num_agent_features=modular_agent_features, num_item_features=modular_item_features)
        .with_num_simulations(num_simulations)
        .with_sigma(sigma)
        .with_capacity_config(mean_multiplier=0.5, lower_multiplier=0.85, upper_multiplier=1.15)  # Match manual
        .build()
    )

    # Use default theta_star (all ones)
    theta_star = np.ones(num_features)
    
    # Prepare with theta_star to generate bundles
    prepared = scenario.prepare(comm=COMM, timeout_seconds=timeout, seed=seed, theta=theta_star)

    bc = BundleChoice()
    prepared.apply(bc, comm=COMM, stage="estimation")

    metadata = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "sigma": sigma,
    }

    return run_methods("linear_knapsack", bc, theta_star, metadata, timeout)


def setup_quadratic_knapsack(timeout: Optional[int], seed: Optional[int] = None) -> BenchmarkResult:
    num_agents = 20
    num_items = 10
    num_simulations = 1
    agent_mod_dim = 2
    item_mod_dim = 1
    agent_quad_dim = 1
    item_quad_dim = 1
    num_features = agent_mod_dim + agent_quad_dim + item_mod_dim + item_quad_dim
    sigma = 1.0

    # Generate data manually (no factory yet for quadratic knapsack)
    rng = np.random.default_rng(seed)
    
    if IS_ROOT:
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
        errors = sigma * rng.normal(scale=1.0, size=(num_agents, num_items))
        estimation_errors = sigma * rng.normal(scale=1.0, size=(num_simulations, num_agents, num_items))
        
        generation_data = {
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
    else:
        generation_data = None

    # Create config
    config = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simulations": num_simulations,
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

    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(generation_data if IS_ROOT else None)
    bc.oracles.build_from_data()
    
    # Generate observed bundles with timeout protection
    theta_star = np.ones(num_features)
    with maybe_timeout(timeout):
        obs_bundles = bc.subproblems.init_and_solve(theta_star)
    
    if IS_ROOT:
        estimation_data = {
            "agent_data": generation_data["agent_data"],
            "item_data": generation_data["item_data"],
            "errors": estimation_errors,
            "obs_bundle": obs_bundles,
        }
    else:
        estimation_data = None

    bc.load_config(config)
    bc.data.load_and_scatter(estimation_data if IS_ROOT else None)
    bc.oracles.build_from_data()
    bc.subproblems.load()

    metadata = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "sigma": sigma,
    }

    return run_methods("quadratic_knapsack", bc, theta_star, metadata, timeout)


def setup_supermod(timeout: Optional[int], seed: Optional[int] = None) -> BenchmarkResult:
    num_agents = 1000
    num_items = 100
    num_simulations = 1
    modular_agent_features = 5
    quadratic_item_features = 1
    num_features = modular_agent_features + quadratic_item_features
    sigma = 5.0

    # Use factory to generate data (matches manual generation)
    scenario = (
        ScenarioLibrary.quadratic_supermodular()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(
            num_mod_agent=modular_agent_features,
            num_mod_item=0,
            num_quad_agent=0,
            num_quad_item=quadratic_item_features,
        )
        .with_num_simulations(num_simulations)
        .with_sigma(sigma)
        .with_agent_modular_config(multiplier=-5.0, mean=0.0, std=1.0)  # Match manual: -5 * abs(normal(0,1))
        .with_quadratic_method(
            method=QuadraticGenerationMethod.BINARY_CHOICE,
            binary_prob=0.2,
            binary_value=1.0,
        )  # Match manual: random.choice([0,1], p=[0.8, 0.2])
        .build()
    )

    prepared = scenario.prepare(comm=COMM, timeout_seconds=timeout, seed=seed)
    theta_star = prepared.theta_star

    # Override config to match manual (theta_lbs for supermodularity)
    prepared.config["row_generation"]["theta_lbs"] = [None] * modular_agent_features + [0.0]

    bc = BundleChoice()
    prepared.apply(bc, comm=COMM, stage="estimation")

    metadata = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "sigma": sigma,
    }

    return run_methods("quadratic_supermodular", bc, theta_star, metadata, timeout)


SCENARIO_RUNNERS: Dict[str, Callable[[Optional[int], Optional[int]], BenchmarkResult]] = {
    "greedy": setup_greedy,
    "plain_single_item": setup_plain_single_item,
    "linear_knapsack": setup_knapsack,
    "quadratic_knapsack": setup_quadratic_knapsack,
    "quadratic_supermodular": setup_supermod,
}


def write_results(results: Tuple[BenchmarkResult, ...]):
    if not IS_ROOT:
        return

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    file_exists = RESULTS_PATH.exists()

    with RESULTS_PATH.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].to_row().keys()))
        if not file_exists:
            writer.writeheader()
        for res in results:
            if res.status == "worker":
                continue
            writer.writerow(res.to_row())

    print(f"\nResults appended to {RESULTS_PATH}\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark row vs column generation.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_RUNNERS.keys(),
        default=list(SCENARIO_RUNNERS.keys()),
        help="Subset of scenarios to benchmark (default: all).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional per-scenario timeout in seconds (uses POSIX alarm).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional NumPy random seed (applied on all ranks).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Note: seed is now handled by factory, but we keep this for any remaining manual RNG usage
    if args.seed is not None:
        np.random.seed(args.seed + RANK)

    results = []
    for scenario in args.scenarios:
        if IS_ROOT:
            print(f"\n=== Scenario: {scenario} ===", flush=True)
        barrier()
        try:
            res = SCENARIO_RUNNERS[scenario](args.timeout, args.seed)
        except Exception as exc:  # pylint: disable=broad-except
            barrier()
            if IS_ROOT:
                print(f"✗ Scenario '{scenario}' failed: {exc}", file=sys.stderr, flush=True)
            raise
        if IS_ROOT:
            print(
                f"Status: {res.status}, "
                f"row_time={res.row_time}, column_time={res.column_time}, "
                f"obj_gap={res.objective_gap}, theta_diff={res.theta_norm_diff}",
                flush=True,
            )
        results.append(res)
        barrier()

    write_results(tuple(results))


if __name__ == "__main__":
    main()

