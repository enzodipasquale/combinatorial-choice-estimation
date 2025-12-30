#!/usr/bin/env python3
"""Small test script for quadknapsack experiment - 100 agents, 50 items."""
import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import datetime
from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from bundlechoice.estimation import RowGeneration1SlackSolver


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Small test dimensions - using simpler settings like the working test
    num_agents = 100
    num_items = 50
    num_agent_modular = 1
    num_agent_quadratic = 1
    num_item_modular = 1
    num_item_quadratic = 1
    num_features = num_agent_modular + num_agent_quadratic + num_item_modular + num_item_quadratic
    num_simuls = 1
    sigma = 1.0  # Simpler sigma like working test
    seed = 42
    timeout_seconds = 300  # 5 minutes for test

    if rank == 0:
        print(f"Testing quadknapsack with {num_agents} agents, {num_items} items")
        print(f"  Agent modular features: {num_agent_modular}")
        print(f"  Agent quadratic features: {num_agent_quadratic}")
        print(f"  Item modular features: {num_item_modular}")
        print(f"  Item quadratic features: {num_item_quadratic}")
        print(f"  Total features: {num_features}")
        print("=" * 80)

    # Use factory to generate data - using default settings like the working test
    scenario = (
        ScenarioLibrary.quadratic_knapsack()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(
            num_agent_modular=num_agent_modular,
            num_agent_quadratic=num_agent_quadratic,
            num_item_modular=num_item_modular,
            num_item_quadratic=num_item_quadratic,
        )
        .with_num_simuls(num_simuls)
        .with_sigma(sigma)
        .build()
    )

    # Create custom theta before prepare
    theta_true = np.ones(num_features)
    theta_true[-1] = 0.1  # Custom theta like working test

    if rank == 0:
        print(f"\nTheta_0: {theta_true}")

    # Prepare with custom theta to avoid generating bundles twice
    # This generates bundles internally and returns them in estimation_data
    prepared = scenario.prepare(comm=comm, timeout_seconds=timeout_seconds, seed=seed, theta=theta_true)

    # Get observed bundles from prepare() (already computed with theta_true) - only on rank 0
    if rank == 0:
        obs_bundles = prepared.estimation_data["obs_bundle"]
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

    result_row = bc.row_generation.solve()
    theta_row = result_row.theta_hat

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

    result_row1 = rg1.solve()
    theta_row1 = result_row1.theta_hat

    if rank == 0:
        elapsed = (datetime.now() - tic).total_seconds()
        print(f"✓ Row generation 1slack completed in {elapsed:.4f} seconds")
        print(f"  Estimated parameters: {theta_row1}")
        print(f"  Parameters close: {np.allclose(theta_row, theta_row1, atol=1e-2, rtol=0)}")
        print("\n" + "=" * 80)
        print("Test completed successfully!")


if __name__ == '__main__':
    main()
