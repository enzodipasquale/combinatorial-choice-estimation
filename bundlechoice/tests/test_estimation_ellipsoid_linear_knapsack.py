import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidSolver

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

def test_ellipsoid_linear_knapsack():
    """Test EllipsoidSolver using obs_bundle generated from linear knapsack subproblem manager."""
    num_agents = 100  # Smaller problem for linear knapsack
    num_items = 20
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
            "name": "LinearKnapsack",
            "settings": {
                "capacity": 5,  # Maximum items per bundle
                "weights": None  # Will be set to ones
            }
        },
        "ellipsoid": {
            "max_iterations": 50,  # Fewer iterations for smaller problem
            "tolerance": 1e-5,
            "initial_radius": 1.0,
            "decay_factor": 0.9,
            "min_volume": 1e-10,
            "verbose": True
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
    knapsack_demo = BundleChoice()
    knapsack_demo.load_config(cfg)
    knapsack_demo.data.load_and_scatter(input_data)
    knapsack_demo.features.load(features_oracle)

    # Simulate beta_star and generate obs_bundles
    beta_star = np.ones(num_features)
    obs_bundles = knapsack_demo.subproblems.init_and_solve(beta_star)

    # Estimate parameters using ellipsoid method
    if rank == 0:
        print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
        print(f"aggregate: {obs_bundles.sum()}")
        input_data["obs_bundle"] = obs_bundles
        input_data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    else:
        input_data = None

    knapsack_demo.load_config(cfg)
    knapsack_demo.data.load_and_scatter(input_data)
    knapsack_demo.features.load(features_oracle)
    knapsack_demo.subproblems.load()
    
    theta_hat = knapsack_demo.ellipsoid.solve()
    
    if rank == 0:
        print("theta_hat:", theta_hat)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        # Additional assertions for ellipsoid method
        assert np.all(np.isfinite(theta_hat))
        # Check that the solution is reasonable (not all zeros or extreme values)
        assert np.any(theta_hat != 0)
        assert np.all(np.abs(theta_hat) < 100)  # Reasonable bounds 