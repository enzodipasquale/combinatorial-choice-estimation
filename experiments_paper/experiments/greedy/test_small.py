#!/usr/bin/env python3
"""Small test script for greedy experiment - 100 agents, 50 items."""
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import datetime
from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.estimation import RowGeneration1SlackSolver
from bundlechoice.factory.utils import mpi_call_with_timeout


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Small test dimensions
    num_agents = 100
    num_items = 50
    num_features = 5
    num_simuls = 1
    sigma = 1.0
    seed = 42
    timeout_seconds = 300  # 5 minutes for test

    if rank == 0:
        print(f"Testing greedy with {num_agents} agents, {num_items} items, {num_features} features")
        print("=" * 80)

    # Use factory to generate data (matches benchmarking/greedy/experiment.py)
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_num_features(num_features)
        .with_num_simuls(num_simuls)
        .with_sigma(sigma)
        .build()
    )

    # Create custom theta before prepare
    theta_true = np.ones(num_features)
    theta_true[-1] = 0.1  # Custom theta for greedy (matches benchmarking)

    # Prepare with custom theta to avoid generating bundles twice
    # This generates bundles internally and returns them in estimation_data
    def prepare_scenario():
        return scenario.prepare(comm=comm, timeout_seconds=timeout_seconds, seed=seed, theta=theta_true)

    prepared = mpi_call_with_timeout(
        comm, prepare_scenario, timeout_seconds, "greedy-prep-test"
    )

    # Get observed bundles from prepare() (already computed with theta_true) - only on rank 0
    if rank == 0:
        obs_bundles = prepared.estimation_data["obs_bundle"]
        print(f"  Aggregate demands: {obs_bundles.sum(1).min()} to {obs_bundles.sum(1).max()}")
        print(f"  Total aggregate: {obs_bundles.sum()}")
    else:
        obs_bundles = None

    # Create BundleChoice for estimation (only when needed)
    bc = BundleChoice()
    prepared.apply(bc, comm=comm, stage="estimation")
    bc.subproblems.load()

    # Run row generation
    if rank == 0:
        print("\nRunning row generation...")
        tic = datetime.now()

    def solve_row_gen():
        return bc.row_generation.solve()

    theta_row = mpi_call_with_timeout(
        comm, solve_row_gen, timeout_seconds, "greedy-rg-test"
    )

    if rank == 0:
        elapsed = (datetime.now() - tic).total_seconds()
        print(f"✓ Row generation completed in {elapsed:.4f} seconds")
        print(f"  Estimated parameters: {theta_row}")
        print(f"  True parameters: {theta_true}")

    # Run 1slack
    if rank == 0:
        print("\nRunning row generation 1slack...")
        tic = datetime.now()

    rg1 = RowGeneration1SlackSolver(
        comm_manager=bc.comm_manager,
        dimensions_cfg=bc.config.dimensions,
        row_generation_cfg=bc.config.row_generation,
        data_manager=bc.data_manager,
        feature_manager=bc.feature_manager,
        subproblem_manager=bc.subproblem_manager,
    )

    def solve_row_gen_1slack():
        return rg1.solve()

    theta_row1 = mpi_call_with_timeout(
        comm, solve_row_gen_1slack, timeout_seconds, "greedy-rg1-test"
    )

    if rank == 0:
        elapsed = (datetime.now() - tic).total_seconds()
        print(f"✓ Row generation 1slack completed in {elapsed:.4f} seconds")
        print(f"  Estimated parameters: {theta_row1}")
        print(f"  Parameters close: {np.allclose(theta_row, theta_row1, atol=1e-2, rtol=0)}")
        print("\n" + "=" * 80)
        print("Test completed successfully!")


if __name__ == '__main__':
    main()

