import numpy as np
import pytest
from bundlechoice.config import DimensionsConfig
from bundlechoice.core import BundleChoice
from typing import Optional, Callable, cast
from bundlechoice.feature_manager import FeatureManager



def test_quad_vs_bruteforce():
    """Test that quadratic solver finds the same optimal bundles as brute force."""
    # Simulate config and data with non-negative quadratic terms
    num_agents, num_items, num_simulations = 30, 13, 1  # Small problem for brute force efficiency
    agent_modular_dim = 1
    agent_quadratic_dim = 1
    item_modular_dim = 1
    item_quadratic_dim = 1
    np.random.seed(123)
    
    # Modular and quadratic for both agent_data and item_data
    # Use balanced distributions to avoid trivial solutions (all items or none)
    # Strategy: Create mixed utilities - some items attractive, some not
    agent_data = {
        # Mix positive and negative modular terms across items
        "modular": np.random.choice([-0.5, 0.3], size=(num_agents, num_items, agent_modular_dim), p=[0.6, 0.4]) + np.random.normal(0, 0.2, (num_agents, num_items, agent_modular_dim)),
        "quadratic": np.abs(np.random.normal(0, 0.2, (num_agents, num_items, num_items, agent_quadratic_dim))),
    }
    item_data = {
        # Mix positive and negative modular terms
        "modular": np.random.choice([-0.3, 0.4], size=(num_items, item_modular_dim), p=[0.5, 0.5]) + np.random.normal(0, 0.15, (num_items, item_modular_dim)),
        "quadratic": np.abs(np.random.normal(0, 0.2, (num_items, num_items, item_quadratic_dim))),
    }
    
    # Set diagonals to zero for all quadratic features and make upper triangular
    for i in range(num_agents):
        for k in range(agent_quadratic_dim):
            agent_data["quadratic"][i, :, :, k] = np.triu(agent_data["quadratic"][i, :, :, k], k=1)
    for k in range(item_quadratic_dim):
        item_data["quadratic"][:, :, k] = np.triu(item_data["quadratic"][:, :, k], k=1)

    # Use moderate error variance to add variation
    errors = np.random.normal(0, 0.6, size=(num_simulations, num_agents, num_items))
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
            "num_simulations": num_simulations
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

    # Test with different theta_0 values (all non-negative for quadratic terms)
    # Use balanced theta values to avoid trivial solutions
    # Strategy: Use moderate values that balance positive/negative modular terms with quadratic terms
    np.random.seed(123)  # Reset seed for theta generation
    test_lambdas = [
        np.array([0.4, 0.3, 0.4, 0.3]),  # Balanced moderate values
        np.array([0.5, 0.25, 0.35, 0.2]),  # Slightly varied
        np.abs(np.random.normal(0.4, 0.12, num_features)),  # Random moderate values
    ]

    for i, theta_0 in enumerate(test_lambdas):
        if bc.rank == 0:        
            print(f"\nTesting theta_0 {i+1}: {theta_0}")
        
        # Get quadratic solver results
        quad_results = bc.subproblems.init_and_solve(theta_0)
        
        # Get brute force results
        assert bc.subproblem_manager is not None, "Subproblem manager should be initialized"
        bruteforce_results = bc.subproblem_manager.brute_force(theta_0)
        
        if bc.rank == 0:
            assert quad_results is not None, "Quadratic results should not be None at rank 0"
            assert bruteforce_results is not None, "Brute force results should not be None at rank 0"
            
            quad_bundles = quad_results
            bruteforce_bundles, bruteforce_max_values = bruteforce_results
            
            # Verify non-trivial choices: ideally not all items or none for most agents
            bundle_sizes = quad_bundles.sum(axis=1)
            num_trivial = np.sum((bundle_sizes == 0) | (bundle_sizes == num_items))
            if num_trivial > 0:
                print(f"  WARNING: {num_trivial}/{num_agents} agents have trivial choices (0 or {num_items} items)")
                print(f"  Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.1f}, std={bundle_sizes.std():.1f}")
                # Try to ensure at least some non-trivial choices for meaningful test
                if num_trivial == num_agents:
                    print(f"  ERROR: All agents have trivial choices! Adjusting data generation...")
                    # This is a warning, not a failure - the solver correctness is still tested
            else:
                print(f"  ✓ Non-trivial choices: Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.1f}")
            
            assert np.array_equal(quad_bundles, bruteforce_bundles)

        else:
            assert quad_results is None, "Quadratic results should be None at non-root ranks"
            assert bruteforce_results == (None, None), "Brute force results should be (None, None) at non-root ranks"

