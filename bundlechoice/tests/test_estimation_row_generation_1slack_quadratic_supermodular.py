import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

def test_row_generation_1slack_quadsupermodular():
    """Test RowGeneration1SlackSolver using observed bundles generated from quadsupermodular subproblem manager."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    num_agents = 20
    num_items = 50
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_quadratic_agent_features = 0
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
    num_simuls = 1
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "QuadSupermodularNetwork",
            "settings": {}
        },
        "row_generation": {
            "max_iters": 200,
            "tolerance_optimality": 0.0005,
            "min_iters": 1,
            "gurobi_settings": {
                "OutputFlag": 0
            }
        }
    }
    
    # Generate data on rank 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        agent_modular = -2 * np.abs(np.random.normal(2, 1, (num_agents, num_items, num_modular_agent_features)))
        item_modular = -2 * np.abs(np.random.normal(2, 1, (num_items, num_modular_item_features)))
        item_quadratic = 1 * np.exp(-np.abs(np.random.normal(0, 1, (num_items, num_items, num_quadratic_item_features))))
        
        for k in range(num_quadratic_item_features):
            np.fill_diagonal(item_quadratic[:, :, k], 0)
            # Multiply by binary matrix with density .1
            item_quadratic[:, :, k] *= (np.random.rand(num_items, num_items) < .3)

        input_data = {
            "item_data": {
                "modular": item_modular,
                "quadratic": item_quadratic
            },
            "agent_data": {
                "modular": agent_modular,
            },
            "errors": 5 * np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
        }
    else:
        input_data = None
        
    quad_demo = BundleChoice()
    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    
    theta_0 = np.ones(num_features)
    observed_bundles = quad_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        total_demand = observed_bundles.sum(1)
        print("Demand range:", total_demand.min(), total_demand.max())
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = 5 * np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    else:
        input_data = None

    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    quad_demo.subproblems.load()
    
    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackSolver(
        comm_manager=quad_demo.comm_manager,
        dimensions_cfg=quad_demo.config.dimensions,
        row_generation_cfg=quad_demo.config.row_generation,
        data_manager=quad_demo.data_manager,
        feature_manager=quad_demo.feature_manager,
        subproblem_manager=quad_demo.subproblem_manager
    )
    theta_hat = solver.solve()
    
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        
        # Check parameter recovery - should be close to true parameters
        param_error = np.linalg.norm(theta_hat - theta_0)
        print(f"Parameter recovery error (L2 norm): {param_error:.6f}")
        assert param_error < 2.5, f"Parameter recovery error too large: {param_error}"
        
        # Check relative error for each parameter
        relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
        max_relative_error = np.max(relative_errors)
        print(f"Max relative error: {max_relative_error:.6f}")
        assert max_relative_error < 1.5, f"Max relative error too large: {max_relative_error}"
