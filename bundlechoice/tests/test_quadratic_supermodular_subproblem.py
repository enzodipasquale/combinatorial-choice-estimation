import numpy as np
import pytest
from bundlechoice.config import DimensionsConfig
from bundlechoice.core import BundleChoice
from typing import Optional, Callable, cast
from bundlechoice.feature_manager import FeatureManager



def test_quad_vs_bruteforce():
    """Test that quadratic solver finds the same optimal bundles as brute force."""
    # Simulate config and data with non-negative quadratic terms
    num_agents, num_items, num_simuls = 100, 13, 1  # Small problem for brute force efficiency
    agent_modular_dim = 1
    agent_quadratic_dim = 1
    item_modular_dim = 1
    item_quadratic_dim = 1
    np.random.seed(123)
    
    # Modular and quadratic for both agent_data and item_data
    agent_data = {
        "modular": np.random.normal(0, 1, (num_agents, num_items, agent_modular_dim)),
        "quadratic": np.abs(np.random.normal(0, 1, (num_agents, num_items, num_items, agent_quadratic_dim))),
    }
    item_data = {
        "modular": np.random.normal(0, 1, (num_items, item_modular_dim)),
        "quadratic": np.abs(np.random.normal(0, 1, (num_items, num_items, item_quadratic_dim))),
    }
    
    # Set diagonals to zero for all quadratic features
    for i in range(num_agents):
        for k in range(agent_quadratic_dim):
            np.fill_diagonal(agent_data["quadratic"][i, :, :, k], 0)
    for k in range(item_quadratic_dim):
        np.fill_diagonal(item_data["quadratic"][:, :, k], 0)

    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors
    }
    
    num_features = agent_modular_dim + agent_quadratic_dim + item_modular_dim + item_quadratic_dim
    
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
    
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.build_from_data()

    # Test with different theta values (all non-negative for quadratic terms)
    test_lambdas = [
        np.ones(num_features),  # All ones
        np.abs(np.random.normal(0, 1, num_features)),  # Random non-negative (absolute values)
        np.array([1] * (num_features - 1) + [0.1]),  # All ones except last is 0.1
    ]

    for i, theta in enumerate(test_lambdas):
        if bc.rank == 0:        
            print(f"\nTesting theta {i+1}: {theta}")
        
        # Get quadratic solver results
        quad_results = bc.subproblems.init_and_solve(theta)
        
        # Get brute force results
        assert bc.subproblem_manager is not None, "Subproblem manager should be initialized"
        bruteforce_results = bc.subproblem_manager.brute_force(theta)
        
        if bc.rank == 0:
            assert quad_results is not None, "Quadratic results should not be None at rank 0"
            assert bruteforce_results is not None, "Brute force results should not be None at rank 0"
            
            quad_bundles = quad_results
            bruteforce_bundles, bruteforce_max_values = bruteforce_results
            
            assert np.array_equal(quad_bundles, bruteforce_bundles)

        else:
            assert quad_results is None, "Quadratic results should be None at non-root ranks"
            assert bruteforce_results == (None, None), "Brute force results should be (None, None) at non-root ranks"

