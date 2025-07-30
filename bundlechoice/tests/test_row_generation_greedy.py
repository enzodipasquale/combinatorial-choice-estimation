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

    modular_agent = np.atleast_2d(modular_agent)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        agent_sum = modular_agent.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, neg_sq))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features

def test_row_generation_greedy():
    """Test RowGenerationSolver using obs_bundle generated from greedy subproblem manager."""
    num_agents = 300
    num_items = 50
    num_features = 6
    num_simuls = 1
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls,
        },
        "subproblem": {
            "name": "Greedy",
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
        modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items)) 
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None

    # Initialize BundleChoice
    greedy_demo = BundleChoice()
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.load(features_oracle)

    # Simulate beta_star and generate obs_bundles
    beta_star = np.ones(num_features)
    obs_bundles = greedy_demo.subproblems.init_and_solve(beta_star)

    # Estimate parameters using row generation
    if rank == 0:
        print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
        print(f"aggregate: {obs_bundles.sum()}")
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    else:
        input_data = None

    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.load(features_oracle)
    greedy_demo.subproblems.load()
    
    lambda_k_iter = greedy_demo.row_generation.solve()
    
    if rank == 0:
        print("lambda_k_iter:", lambda_k_iter)
        assert lambda_k_iter.shape == (num_features,)
        assert not np.any(np.isnan(lambda_k_iter)) 

