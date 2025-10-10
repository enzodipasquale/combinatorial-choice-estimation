import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver

def test_row_generation_1slack_linear_knapsack():
    """Test RowGeneration1SlackSolver using observed bundles generated from linear knapsack subproblem manager."""
    num_agents = 20
    num_items = 20
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "LinearKnapsack",
            "settings": {"TimeLimit": 10, "MIPGap_tol": 0.01}
        },
        "row_generation": {
            "max_iters": 100,
            "tolerance_optimality": 0.0001,
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
        input_data = {
            "item_data": {
                "modular": np.abs(np.random.normal(0, 1, (num_items, num_modular_item_features))),
                "weights": np.random.randint(1, 10, size=num_items)
            },
            "agent_data": {
                "modular": np.abs(np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))),
                "capacity": np.random.randint(1, 100, size=num_agents)
            },
            "errors": np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
        }
    else:
        input_data = None
        
    knapsack_demo = BundleChoice()
    knapsack_demo.load_config(cfg)
    knapsack_demo.data.load_and_scatter(input_data)
    knapsack_demo.features.build_from_data()
    
    theta_0 = np.ones(num_features)
    observed_bundles = knapsack_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        print("Total demand:", observed_bundles.sum(1).min(), observed_bundles.sum(1).max())
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    else:
        input_data = None

    knapsack_demo.load_config(cfg)
    knapsack_demo.data.load_and_scatter(input_data)
    knapsack_demo.features.build_from_data()
    knapsack_demo.subproblems.load()

    # Use 1slack solver instead of regular row generation
    solver = RowGeneration1SlackSolver(
        comm_manager=knapsack_demo.comm_manager,
        dimensions_cfg=knapsack_demo.config.dimensions,
        row_generation_cfg=knapsack_demo.config.row_generation,
        data_manager=knapsack_demo.data_manager,
        feature_manager=knapsack_demo.feature_manager,
        subproblem_manager=knapsack_demo.subproblem_manager
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
        assert param_error < 1.0, f"Parameter recovery error too large: {param_error}"
        
        # Check relative error for each parameter
        relative_errors = np.abs(theta_hat - theta_0) / (np.abs(theta_0) + 1e-8)
        max_relative_error = np.max(relative_errors)
        print(f"Max relative error: {max_relative_error:.6f}")
        assert max_relative_error < 0.5, f"Max relative error too large: {max_relative_error}"
