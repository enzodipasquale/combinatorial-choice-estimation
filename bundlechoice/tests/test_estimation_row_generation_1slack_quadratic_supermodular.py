import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackManager
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.scenarios.data_generator import QuadraticGenerationMethod


def test_row_generation_1slack_quadsupermodular():
    """Test RowGeneration1SlackSolver using observed bundles generated from quadsupermodular subproblem manager."""
    seed = 42
    
    num_agents = 20
    num_items = 50
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_quadratic_agent_features = 0
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
    num_simulations = 1
    sigma = 5.0
    
    # Use factory to generate data (matches manual: -2*abs(normal(2,1)), exponential quadratic)
    scenario = (
        ScenarioLibrary.quadratic_supermodular()
        .with_dimensions(num_agents=num_agents, num_items=num_items)
        .with_feature_counts(
            num_mod_agent=num_modular_agent_features,
            num_mod_item=num_modular_item_features,
            num_quad_agent=num_quadratic_agent_features,
            num_quad_item=num_quadratic_item_features,
        )
        .with_num_simulations(num_simulations)
        .with_sigma(sigma)
        .with_agent_modular_config(multiplier=-2.0, mean=2.0, std=1.0)
        .with_quadratic_method(
            method=QuadraticGenerationMethod.EXPONENTIAL,
            mask_threshold=0.3,
        )  # Matches manual: exp(-abs(normal)) with 0.3 mask
        .build()
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed)
    theta_0 = prepared.theta_star
        
    quad_demo = BundleChoice()
    prepared.apply(quad_demo, comm=comm, stage="generation")
    
    observed_bundles = quad_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        total_demand = observed_bundles.sum(1)
        print("Demand range:", total_demand.min(), total_demand.max())

    # Apply estimation data
    prepared.apply(quad_demo, comm=comm, stage="estimation")
    quad_demo.subproblems.load()
    
    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackManager(
        comm_manager=quad_demo.comm_manager,
        dimensions_cfg=quad_demo.config.dimensions,
        row_generation_cfg=quad_demo.config.row_generation,
        data_manager=quad_demo.data_manager,
        oracles_manager=quad_demo.oracles_manager,
        subproblem_manager=quad_demo.subproblem_manager
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
        # Allow slightly higher tolerance for 1slack method which may be less accurate
        assert param_error < 3.5, f"Parameter recovery error too large: {param_error}"
        
        # Check relative error for each parameter
        relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
        max_relative_error = np.max(relative_errors)
        print(f"Max relative error: {max_relative_error:.6f}")
        # Allow higher tolerance for 1slack method which may be less accurate
        assert max_relative_error < 3.0, f"Max relative error too large: {max_relative_error}"
