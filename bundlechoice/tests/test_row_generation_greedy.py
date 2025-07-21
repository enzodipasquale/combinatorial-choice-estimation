import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver

def features_oracle(i_id, B_j, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_item = data["item_data"]["modular"]

    modular_agent = np.atleast_2d(modular_agent)
    modular_item = np.atleast_2d(modular_item)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True

    agent_sum = modular_agent.T @ B_j
    item_sum = modular_item.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, item_sum, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features

def test_row_generation_greedy():
    """Test RowGenerationSolver using obs_bundle generated from greedy subproblem manager."""
    num_agents = 300
    num_items = 50
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = 5
    num_simuls = 1
    # np.random.seed(42)
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "Greedy",
            "settings": {}
        }
    }
    input_data = {
        "item_data": {"modular": np.abs(np.random.normal(0, 1, (num_items, num_modular_item_features)))},
        "agent_data": {"modular": np.abs(np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features)))},
        "errors": np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
    }

    # Generate obs_bundles
    greedy_demo = BundleChoice()
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.load(features_oracle)

    lambda_k = np.ones(num_features)
    lambda_k[-1] = 0.1
    obs_bundle = greedy_demo.subproblems.init_and_solve(lambda_k)
    if greedy_demo.rank == 0:
        print(obs_bundle.sum(1))

    # Estimate parameters
    input_data["obs_bundle"] = obs_bundle
    input_data["errors"] = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    rowgen_cfg = {
        "max_iters": 100,
        "tol_certificate": 0.001,
        "min_iters": 1
    }
    cfg["rowgen"] = rowgen_cfg
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    lambda_k_iter, p_j_iter = greedy_demo.row_generation.solve()
    if greedy_demo.rank == 0:
        print(lambda_k_iter)
        print(p_j_iter) 

