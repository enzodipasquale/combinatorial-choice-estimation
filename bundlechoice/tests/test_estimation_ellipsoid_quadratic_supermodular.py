import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation.ellipsoid import EllipsoidSolver

def test_ellipsoid_quadsupermodular():
    """Test EllipsoidSolver using observed bundles generated from quadsupermodular subproblem manager."""
    num_agents = 100
    num_items = 30
    num_modular_agent_features = 2
    num_modular_item_features = 2
    num_quadratic_agent_features = 0
    num_quadratic_item_features = 2
    num_features = num_modular_agent_features + num_modular_item_features + num_quadratic_agent_features + num_quadratic_item_features
    num_simuls = 1
    sigma = 5
    
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
        },
        "ellipsoid": {
            "num_iters": 50,
            "initial_radius": 20 * np.sqrt(num_features),
            "verbose": False
        }
    }
    
    # Generate data on rank 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        agent_modular = -2 * np.abs(np.random.normal(2, 1, (num_agents, num_items, num_modular_agent_features)))
        item_modular = -2 * np.abs(np.random.normal(2, 1, (num_items, num_modular_item_features)))
        item_quadratic = 1 * np.exp(-np.abs(np.random.normal(0, 1, (num_items, num_items, num_quadratic_item_features))))
        errors =  np.random.normal(0, 1, (num_agents, num_items))
        estimation_errors = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
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
            },
            "errors": sigma * errors,
        }
    else:
        input_data = None
        
    quad_demo = BundleChoice()
    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    
    theta_0 = np.ones(num_features)
    observed_bundles = quad_demo.subproblems.init_and_solve(theta_0)
    
    if rank == 0 and observed_bundles is not None:
        total_demand = observed_bundles.sum(1)
        print("Demand range:", total_demand.min(), total_demand.max())
        input_data["obs_bundle"] = observed_bundles
        input_data["errors"] = sigma * estimation_errors
    else:
        input_data = None

    quad_demo.load_config(cfg)
    quad_demo.data.load_and_scatter(input_data)
    quad_demo.features.build_from_data()
    quad_demo.subproblems.load()
    
    theta_hat = quad_demo.ellipsoid.solve()
    
    # Check objective values on all ranks
    obj_at_theta_0 = quad_demo.ellipsoid.objective(theta_0)
    obj_at_theta_hat = quad_demo.ellipsoid.objective(theta_hat)
    
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
        
        print("obj_at_theta_0", obj_at_theta_0)
        print("obj_at_theta_hat", obj_at_theta_hat) 