import numpy as np
import pytest
import time
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidSolver
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem



def test_ellipsoid_plain_single_item():
    """Test EllipsoidSolver using PlainSingleItemSubproblem with only modular features."""
    num_agents = 1000
    num_items = 100
    num_modular_agent_features = 10
    num_modular_item_features = 1
    num_features = num_modular_agent_features + num_modular_item_features
    num_simuls = 1
    sigma = 1


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
        "ellipsoid": {
            "num_iters": 150,
            "initial_radius": 20 * np.sqrt(num_features),
            "verbose": False
        }
    }
    
    # Generate data on rank 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        errors = sigma * np.random.normal(0, 1, ( num_agents, num_items))
        estimation_errors = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
        input_data = {
            "item_data": {"modular": np.random.normal(0, 1, (num_items, num_modular_item_features))},
            "agent_data": {"modular": np.random.normal(0, 1, (num_agents, num_items, num_modular_agent_features))},
            "errors": errors,
            "estimation_errors": estimation_errors,
        }
    else:
        input_data = None
        
    demo = BundleChoice()
    demo.load_config(cfg)
    demo.data.load_and_scatter(input_data)
    demo.features.build_from_data()
    
    theta_0 = np.ones(num_features)
    observed_bundles = demo.subproblems.init_and_solve(theta_0)
    
    # Check that observed_bundles is not None
    if rank == 0:
        assert observed_bundles is not None, "observed_bundles is None!"
        assert input_data["errors"] is not None, "input_data['errors'] is None!"
        assert observed_bundles.shape == (num_agents, num_items)
        assert np.all(observed_bundles.sum(axis=1) <= 1), "Each agent should select at most one item."
        no_selection = np.where(observed_bundles.sum(axis=1) == 0)[0]

        modular_agent = input_data["agent_data"]["modular"]
        modular_item = input_data["item_data"]["modular"]
        errors = input_data["errors"]
        agent_util = np.einsum('aij,j->ai', modular_agent, theta_0[:num_modular_agent_features])
        item_util = np.dot(modular_item, theta_0[num_modular_agent_features:])
        total_util = agent_util + item_util + errors
        j_star = np.argmax(total_util, axis=1)
        for i in range(num_agents):
            if observed_bundles[i, :].sum() == 1:
                assert observed_bundles[i, j_star[i]] == 1, f"Agent {i} did not select the max utility item."
            else:
                assert np.all(total_util[i, :] <= 0), f"Agent {i} made no selection, but has positive utility: {total_util[i, :]}"
        print("theta_0:\n", theta_0)
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = estimation_errors
    else:
        input_data = None

    demo.load_config(cfg)
    demo.data.load_and_scatter(input_data)
    demo.features.build_from_data()
    demo.subproblems.load()
    
    tic = time.time()
    theta_hat = demo.ellipsoid.solve()
    toc = time.time()
    
    # # Check objective values on all ranks
    # obj_at_theta_0 = demo.ellipsoid.objective(theta_0)
    # obj_at_theta_hat = demo.ellipsoid.objective(theta_hat)
    
    # if rank == 0:
    #     assert theta_hat.shape == (num_features,)
    #     assert not np.any(np.isnan(theta_hat))
    #     # Additional assertions for ellipsoid method
    #     assert np.all(np.isfinite(theta_hat))
    #     # Check that the solution is reasonable (not all zeros or extreme values)
    #     assert np.any(theta_hat != 0)
    #     assert np.all(np.abs(theta_hat) < 100)  # Reasonable bounds
        