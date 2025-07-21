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
    np.random.seed(321)
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
        }
    }
    agent_modular = -2*  np.abs(np.random.normal(2, 1, (num_agents, num_items, num_modular_agent_features)))
    # # agent_quadratic = .5 * np.exp(-np.abs(np.random.normal(0, 1, (num_agents, num_items, num_items, num_quadratic_agent_features))))
    # for i in range(num_agents):
    #     for k in range(num_quadratic_agent_features):
    #         np.fill_diagonal(agent_quadratic[i, :, :, k], 0)
    #         # Multiply by binary matrix with density .1
    #         agent_quadratic[i, :, :, k] *= (np.random.rand(num_items, num_items) < 1)

    item_modular = - 2* np.abs(np.random.normal(2, 1, (num_items, num_modular_item_features)))
    item_quadratic = 1* np.exp(-np.abs(np.random.normal(0, 1, (num_items, num_items, num_quadratic_item_features))))
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
            # "quadratic": agent_quadratic
        },
        "errors": 5 *  np.random.normal(0, 1, (num_simuls, num_agents, num_items)),
    }
    quad_demo = BundleChoice()
    quad_demo.load_config(cfg)
    quad_demo.load_data(input_data, scatter=True)
    quad_demo.build_feature_oracle_from_data()
    lambda_k = np.ones(num_features)
    obs_bundle = quad_demo.init_and_solve_subproblems(lambda_k)
    input_data["obs_bundle"] = obs_bundle
    if quad_demo.rank == 0 and input_data["obs_bundle"] is not None:
        total_demand = input_data["obs_bundle"].sum(1)
        print("Total demand")
        print(total_demand.min())
        print(total_demand.max())
        


    num_simuls = 1
    cfg["dimensions"]["num_simuls"] = num_simuls
    input_data["errors"] = 5 *  np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    # Add rowgen config for the row generation solver
    rowgen_cfg = {
        "max_iters": 100,
        "tol_certificate": 0.001,
        "min_iters": 1
    }
    cfg["rowgen"] = rowgen_cfg
    quad_demo.load_config(cfg)
    quad_demo.load_data(input_data, scatter=True)
    lambda_k_iter, p_j_iter = quad_demo.compute_estimator_row_gen()
    if quad_demo.rank == 0:
        print(lambda_k_iter)
        print(p_j_iter) 
