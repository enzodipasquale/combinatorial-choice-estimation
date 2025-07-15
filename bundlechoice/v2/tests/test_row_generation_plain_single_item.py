import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.v2.core import BundleChoice
from bundlechoice.v2.compute_estimator.row_generation import RowGenerationSolver
from bundlechoice.v2.subproblems.registry.plain_single_item import PlainSingleItemSubproblem


def test_row_generation_plain_single_item():
    """Test RowGenerationSolver using PlainSingleItemSubproblem with only modular features."""
    num_agents = 10000
    num_items = 50
    num_modular_agent_features = 5
    num_modular_item_features = 5
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    np.random.seed(1234)
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
        }
    }
    input_data = {
        "item_data": {"modular": np.random.normal(0, 1, (num_items, num_modular_item_features))},
        "agent_data": {"modular": np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))},
        "errors": np.random.normal(0, 0.1, (num_simuls, num_agents, num_items)),
    }
    demo = BundleChoice()
    demo.load_config(cfg)
    demo.load_data(input_data, scatter=True)
    demo.build_feature_oracle_from_data()
    lambda_k = np.ones(num_features)
    obs_bundle = demo.init_and_solve_subproblems(lambda_k)
    # Check that obs_bundle is not None
    if demo.rank == 0:
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

    num_simuls = 1
    cfg["dimensions"]["num_simuls"] = num_simuls
    input_data["errors"] = np.random.normal(0, 0.1, (num_simuls, num_agents, num_items))
    input_data["obs_bundle"] = obs_bundle
    demo.load_config(cfg)
    demo.load_data(input_data, scatter=True)
    solver = RowGenerationSolver(demo)
    lambda_k_iter, p_j_iter = solver.compute_estimator_row_gen()
    if demo.rank == 0:
        print("lambda_k_iter (row generation result):\n", lambda_k_iter)
        print("p_j_iter (row generation result):\n", p_j_iter)
        assert lambda_k_iter.shape == (num_features,) 
    
    