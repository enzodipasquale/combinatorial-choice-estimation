import numpy as np
import pytest
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem


def test_row_generation_plain_single_item():
    """Test RowGenerationSolver using PlainSingleItemSubproblem with only modular features."""
    num_agents = 2_000
    num_items = 100
    num_modular_agent_features = 4
    num_modular_item_features = 1
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
            "name": "PlainSingleItem",
            "settings": {}
        },
        "rowgen": {
            "max_iters": 100,
            "tol_certificate": .0001,
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
        input_data = {
            "item_data": {"modular": np.random.normal(0, 1, (num_items, num_modular_item_features))},
            "agent_data": {"modular": np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))},
            "errors": np.random.normal(0, 0.1, (num_simuls, num_agents, num_items)),
        }
    else:
        input_data = None
        
    demo = BundleChoice()
    demo.load_config(cfg)
    demo.data.load_and_scatter(input_data)
    demo.features.build_from_data()
    
    lambda_k = np.ones(num_features)
    obs_bundle = demo.subproblems.init_and_solve(lambda_k)
    
    # Check that obs_bundle is not None
    if rank == 0:
        assert obs_bundle is not None, "obs_bundle is None!"
        assert input_data["errors"] is not None, "input_data['errors'] is None!"
        assert obs_bundle.shape == (num_agents, num_items)
        assert np.all(obs_bundle.sum(axis=1) <= 1), "Each agent should select at most one item."
        no_selection = np.where(obs_bundle.sum(axis=1) == 0)[0]

        modular_agent = input_data["agent_data"]["modular"]
        modular_item = input_data["item_data"]["modular"]
        errors = input_data["errors"][0]
        agent_util = np.einsum('aij,j->ai', modular_agent, lambda_k[:num_modular_agent_features])
        item_util = np.dot(modular_item, lambda_k[num_modular_agent_features:])
        total_util = agent_util + item_util + errors
        j_star = np.argmax(total_util, axis=1)
        for i in range(num_agents):
            if obs_bundle[i, :].sum() == 1:
                assert obs_bundle[i, j_star[i]] == 1, f"Agent {i} did not select the max utility item."
            else:
                assert np.all(total_util[i, :] <= 0), f"Agent {i} made no selection, but has positive utility: {total_util[i, :]}"
        print("lambda_k:\n", lambda_k)
        input_data["obs_bundle"] = obs_bundle
        input_data["errors"] = np.random.normal(0, 0.1, (num_simuls, num_agents, num_items))
    else:
        input_data = None

    demo.load_config(cfg)
    demo.data.load_and_scatter(input_data)
    demo.features.build_from_data()
    
    tic = time.time()
    lambda_k_iter = demo.row_generation.solve()
    toc = time.time()
    
    if rank == 0:
        print("lambda_k_iter (row generation result):\n", lambda_k_iter)
        print(f"Time taken: {toc - tic} seconds")
        assert lambda_k_iter.shape == (num_features,)
        assert not np.any(np.isnan(lambda_k_iter)) 
    
    