import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver

def test_row_generation_quadsupermodular():
    """Test RowGenerationSolver using obs_bundle generated from quadsupermodular subproblem manager."""
    num_agents = 500
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
        "rowgen": {
            "max_iters": 100,
            "tol_certificate": 0.001,
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
    
    lambda_k = np.ones(num_features)
    obs_bundle = quad_demo.subproblems.init_and_solve(lambda_k)
    
    if rank == 0 and obs_bundle is not None:
        total_demand = obs_bundle.sum(1)
        print("Total demand")
        print(total_demand.min())
        print(total_demand.max())
        input_data["obs_bundle"] = obs_bundle
        input_data["errors"] = 5 * np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    else:
        input_data = None

    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    quad_demo.subproblems.load()
    
    lambda_k_iter = quad_demo.row_generation.solve()
    
    if rank == 0:
        print("lambda_k_iter:", lambda_k_iter)
        assert lambda_k_iter.shape == (num_features,)
        assert not np.any(np.isnan(lambda_k_iter)) 
