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
    modular_item = data["item_data"]["modular"]

    modular_agent = np.atleast_2d(modular_agent)
    modular_item = np.atleast_2d(modular_item)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        agent_sum = modular_agent.T @ B_j
        item_sum = modular_item.T @ B_j
    features = np.vstack((agent_sum, item_sum))
    if single_bundle:
        return features[:, 0]  # Return as 1D array for a single bundle
    return features

def test_ellipsoid_linear_knapsack():
    """Test EllipsoidSolver using observed bundles generated from linear knapsack subproblem manager."""
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
        "ellipsoid": {
            "num_iters": 100,
            "initial_radius": 20 * np.sqrt(num_features),
            "verbose": False
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
    knapsack_demo.features.set_oracle(features_oracle)
    
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
    knapsack_demo.features.set_oracle(features_oracle)
    knapsack_demo.subproblems.load()

    theta_hat = knapsack_demo.ellipsoid.solve()
    
    # Check objective values on all ranks
    obj_at_theta_0 = knapsack_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = knapsack_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        print("obj_at_theta_0", obj_at_theta_0)
        print("obj_at_theta_hat", obj_at_theta_hat) 
    
