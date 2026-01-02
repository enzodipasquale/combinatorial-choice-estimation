import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager
from bundlechoice.scenarios import ScenarioLibrary


def test_row_generation_1slack_greedy():
    """Test RowGeneration1SlackSolver using observed bundles generated from greedy subproblem manager."""
    num_agents = 250
    num_items = 50
    num_features = 6
    num_simulations = 1
    seed = 42
    
    # Use factory to generate data (matches manual: normal(0,1) without abs)
    scenario = (
        ScenarioLibrary.greedy()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_num_features(num_features)
        .with_num_simulations(num_simulations)
        .with_sigma(1.0)
        .with_agent_config(apply_abs=False)  # Match manual: no abs
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star

    # Initialize BundleChoice
    greedy_demo = BundleChoice()
    prepared.apply(greedy_demo, comm=comm, stage="generation")

    # Generate observed bundles
    observed_bundles = greedy_demo.subproblems.init_and_solve(theta_0)

    # Estimate parameters using row generation 1slack
    if rank == 0:
        print(f"aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        print(f"aggregate: {observed_bundles.sum()}")

    # Apply estimation data
    prepared.apply(greedy_demo, comm=comm, stage="estimation")
    greedy_demo.subproblems.load()
    
    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackManager(
        comm_manager=greedy_demo.comm_manager,
        dimensions_cfg=greedy_demo.config.dimensions,
        row_generation_cfg=greedy_demo.config.row_generation,
        data_manager=greedy_demo.data_manager,
        feature_manager=greedy_demo.feature_manager,
        subproblem_manager=greedy_demo.subproblem_manager
    )
    result = solver.solve()
    
    if rank == 0:
        theta_hat = result.theta_hat
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        
        # Check parameter recovery - should be close to true parameters
        param_error = np.linalg.norm(theta_hat - theta_0)
        print(f"Parameter recovery error (L2 norm): {param_error:.6f}")
        assert param_error < 1.0, f"Parameter recovery error too large: {param_error}"
        
        # Check relative error for each parameter
        relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
        max_relative_error = np.max(relative_errors)
        print(f"Max relative error: {max_relative_error:.6f}")
        assert max_relative_error < 0.5, f"Max relative error too large: {max_relative_error}"
