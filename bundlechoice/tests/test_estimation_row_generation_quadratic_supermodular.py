import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGenerationSolver

def test_row_generation_quadsupermodular():
    """Test RowGenerationSolver using observed bundles generated from quadsupermodular subproblem manager."""
    num_agents = 250
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
            "max_iters": 100,
            "tolerance_optimality": 0.001,
            "min_iters": 1,
            "master_settings": {
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
    
    theta_hat = quad_demo.row_generation.solve()
    
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat)) 
