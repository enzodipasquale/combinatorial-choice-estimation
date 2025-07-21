import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver

def test_row_generation_linear_knapsack():
    """Test RowGenerationSolver using obs_bundle generated from linear knapsack subproblem manager."""
    num_agents = 500
    num_items = 20
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    np.random.seed(123)
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
        }
    }
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
    knapsack_demo = BundleChoice()
    knapsack_demo.load_config(cfg)
    knapsack_demo.load_data(input_data, scatter=True)
    knapsack_demo.build_feature_oracle_from_data()
    lambda_k = np.ones(num_features)
    obs_bundle = knapsack_demo.init_and_solve_subproblems(lambda_k)
    input_data["obs_bundle"] = obs_bundle
    input_data["errors"] = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    if knapsack_demo.rank == 0 and input_data["obs_bundle"] is not None:
        print(input_data["obs_bundle"].sum(1))

    # Add rowgen config for the row generation solver
    rowgen_cfg = {
        "max_iters": 100,
        "tol_certificate": 0.0001,
    }
    cfg["rowgen"] = rowgen_cfg
    knapsack_demo.load_config(cfg)
    knapsack_demo.load_data(input_data, scatter=True)

    lambda_k_iter, p_j_iter = knapsack_demo.compute_estimator_row_gen()
    if knapsack_demo.rank == 0:
        print(lambda_k_iter)
        print(p_j_iter)