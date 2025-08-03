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


def test_ellipsoid_greedy():
    """Test EllipsoidSolver using observed bundles generated from greedy subproblem manager."""
    num_agents = 300
    num_items = 50
    num_features = 4
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
        "ellipsoid": {
            "num_iters": 100,
            "initial_radius": 20 * np.sqrt(num_features),  # Ensure ellipsoid contains vector of all ones with margin
            "verbose": False
        }
    }
    
    # Generate data on rank 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
        errors = np.random.normal(0, 1, size=(num_agents, num_items)) 
        estimation_errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        agent_data = {"modular": modular}
        input_data = {"agent_data": agent_data, "errors": errors}
    else:
        input_data = None

    # Initialize BundleChoice
    greedy_demo = BundleChoice()
    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)

    # Simulate theta_0 and generate observed bundles
    theta_0 = np.ones(num_features)
    observed_bundles = greedy_demo.subproblems.init_and_solve(theta_0)

    # Estimate parameters using ellipsoid method
    if rank == 0:
        print(f"aggregate demands: {observed_bundles.sum(1).min()}, {observed_bundles.sum(1).max()}")
        print(f"aggregate: {observed_bundles.sum()}")
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = estimation_errors
    else:
        input_data = None

    greedy_demo.load_config(cfg)
    greedy_demo.data.load_and_scatter(input_data)
    greedy_demo.features.set_oracle(features_oracle)
    greedy_demo.subproblems.load()
    theta_ = np.ones(num_features) 
    # theta_[0] = .99
    gradient = greedy_demo.ellipsoid.obj_gradient(theta_)
 
    
    theta_hat = greedy_demo.ellipsoid.solve()
    
    if rank == 0:
        print("theta_hat:", theta_hat)
        print("theta_0:", theta_0)
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        # Additional assertions for ellipsoid method
        assert np.all(np.isfinite(theta_hat))
        # Check that the solution is reasonable (not all zeros or extreme values)
        assert np.any(theta_hat != 0)
        assert np.all(np.abs(theta_hat) < 100)  # Reasonable bounds
    obj_at_theta_0 = greedy_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = greedy_demo.ellipsoid.objective(theta_hat)
    if rank == 0:
        print("obj_at_theta_0", obj_at_theta_0)
        print("obj_at_theta_hat", obj_at_theta_hat)

