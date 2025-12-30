import numpy as np
import pytest
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager
from bundlechoice.factory import ScenarioLibrary


def test_row_generation_1slack_plain_single_item():
    """Test RowGeneration1SlackSolver using PlainSingleItemSubproblem with only modular features."""
    num_agents = 500
    num_items = 2
    num_modular_agent_features = 4
    num_modular_item_features = 1
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    sigma = 1
    seed = 42
    
    # Use factory to generate data (no correlation in this test)
    scenario = (
        ScenarioLibrary.plain_single_item()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(num_agent_features=num_modular_agent_features, num_item_features=num_modular_item_features)
        .with_num_simuls(num_simuls)
        .with_sigma(sigma)
        .with_correlation(enabled=False)  # No correlation in this test
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
        
    demo = BundleChoice()
    prepared.apply(demo, comm=comm, stage="generation")
    
    observed_bundles = demo.subproblems.init_and_solve(theta_0)
    
    # Check that observed_bundles is not None
    if rank == 0:
        assert observed_bundles is not None, "observed_bundles is None!"
        assert observed_bundles.shape == (num_agents, num_items)
        assert np.all(observed_bundles.sum(axis=1) <= 1), "Each agent should select at most one item."
        
        # Verify utility maximization
        modular_agent = prepared.generation_data["agent_data"]["modular"]
        modular_item = prepared.generation_data["item_data"]["modular"]
        errors = prepared.generation_data["errors"]
        agent_util = np.einsum('aij,j->ai', modular_agent, theta_0[:num_modular_agent_features])
        item_util = np.dot(modular_item, theta_0[num_modular_agent_features:])
        total_util = agent_util + item_util + errors
        j_star = np.argmax(total_util, axis=1)
        for i in range(num_agents):
            if observed_bundles[i, :].sum() == 1:
                assert observed_bundles[i, j_star[i]] == 1, f"Agent {i} did not select the max utility item."
            else:
                assert np.all(total_util[i, :] <= 0), f"Agent {i} made no selection, but has positive utility: {total_util[i, :]}"
        print("theta_0:\n", theta_0)

    # Apply estimation data
    prepared.apply(demo, comm=comm, stage="estimation")
    demo.subproblems.load()
    
    tic = time.time()
    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackManager(
        comm_manager=demo.comm_manager,
        dimensions_cfg=demo.config.dimensions,
        row_generation_cfg=demo.config.row_generation,
        data_manager=demo.data_manager,
        feature_manager=demo.feature_manager,
        subproblem_manager=demo.subproblem_manager
    )
    result = solver.solve()
    toc = time.time()
    
    if rank == 0:
        theta_hat = result.theta_hat
        print("theta_hat (row generation 1slack result):\n", theta_hat)
        print("theta_0:\n", theta_0)
        print(f"Time taken: {toc - tic} seconds")
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
