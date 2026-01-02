import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager
from bundlechoice.scenarios import ScenarioLibrary


def test_row_generation_1slack_linear_knapsack():
    """Test RowGeneration1SlackSolver using observed bundles generated from linear knapsack subproblem manager."""
    num_agents = 250
    num_items = 20
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    seed = 42
    
    # Use factory to generate data (matches manual: abs(normal), weights 1-10, random capacity 1-100)
    scenario = (
        ScenarioLibrary.linear_knapsack()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(num_agent_features=num_modular_agent_features, num_item_features=num_modular_item_features)
        .with_num_simuls(num_simuls)
        .with_sigma(1.0)
        .with_random_capacity(low=1, high=100)  # Match manual: random capacity
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
        
    knapsack_demo = BundleChoice()
    prepared.apply(knapsack_demo, comm=comm, stage="generation")
    
    observed_bundles = knapsack_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        print("Total demand:", observed_bundles.sum(1).min(), observed_bundles.sum(1).max())

    # Apply estimation data
    prepared.apply(knapsack_demo, comm=comm, stage="estimation")
    knapsack_demo.subproblems.load()

    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackManager(
        comm_manager=knapsack_demo.comm_manager,
        dimensions_cfg=knapsack_demo.config.dimensions,
        row_generation_cfg=knapsack_demo.config.row_generation,
        data_manager=knapsack_demo.data_manager,
        feature_manager=knapsack_demo.feature_manager,
        subproblem_manager=knapsack_demo.subproblem_manager
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
