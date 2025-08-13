#!/usr/bin/env python3
"""
Test to analyze current user experience and identify pain points.
This simulates how a real user would use BundleChoice with their own data.
"""

import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice

def simple_features_oracle(i_id, B_j, data):
    """Simple feature oracle for testing user experience."""
    # Extract agent features
    agent_features = data["agent_data"]["features"][i_id]  # (num_features,)
    
    # Compute features: agent_features * bundle_sum
    bundle_sum = np.sum(B_j, axis=0)  # Sum of bundle items
    features = agent_features * bundle_sum
    
    return features

def test_current_user_experience():
    """
    Test the current user experience to identify pain points.
    This simulates a real user workflow.
    """
    # User's problem setup
    num_agents = 100
    num_items = 20
    num_features = 3
    num_simuls = 1
    
    # User's configuration
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
        "row_generation": {
            "max_iters": 50,
            "tolerance_optimality": 0.001,
            "min_iters": 1,
            "gurobi_settings": {"OutputFlag": 0}
        }
    }
    
    # Simulate user's data (in real usage, this would be loaded from files)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # User's agent data
        agent_features = np.random.normal(0, 1, (num_agents, num_features))
        agent_data = {"features": agent_features}
        
        # User's observed bundles (what they actually observed)
        obs_bundles = np.random.choice([0, 1], size=(num_agents, num_items), p=[0.7, 0.3])
        
        # User's error terms
        errors = np.random.normal(0, 0.1, size=(num_simuls, num_agents, num_items))
        
        input_data = {
            "agent_data": agent_data,
            "obs_bundle": obs_bundles,
            "errors": errors
        }
    else:
        input_data = None

    # === CURRENT USER EXPERIENCE ===
    
    # Step 1: User initializes BundleChoice
    bc = BundleChoice()
    
    # Step 2: User loads configuration
    bc.load_config(cfg)
    
    # Step 3: User loads their data
    bc.data.load_and_scatter(input_data)
    
    # Step 4: User sets up feature computation
    bc.features.set_oracle(simple_features_oracle)
    
    # Step 5: User runs estimation
    theta_hat = bc.row_generation.solve()
    
    # Verify results
    if rank == 0:
        print("=== Current User Experience Test ===")
        print(f"Estimated parameters: {theta_hat}")
        print(f"Parameter shape: {theta_hat.shape}")
        assert theta_hat.shape == (num_features,)
        assert not np.any(np.isnan(theta_hat))
        print("✅ Current UX test passed")

def test_user_experience_pain_points():
    """
    Test to identify specific pain points in the current user experience.
    """
    num_agents = 50
    num_items = 10
    num_features = 2
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {"name": "Greedy"},
        "ellipsoid": {"num_iters": 20, "verbose": False}
    }
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        agent_features = np.random.normal(0, 1, (num_agents, num_features))
        obs_bundles = np.random.choice([0, 1], size=(num_agents, num_items), p=[0.6, 0.4])
        errors = np.random.normal(0, 0.1, size=(num_agents, num_items))
        
        input_data = {
            "agent_data": {"features": agent_features},
            "obs_bundle": obs_bundles,
            "errors": errors
        }
    else:
        input_data = None

    # Test pain points
    bc = BundleChoice()
    
    # Pain Point 1: User has to remember the exact order of operations
    bc.load_config(cfg)
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(simple_features_oracle)
    
    # Pain Point 2: User has to know which solver to call
    theta_hat = bc.ellipsoid.solve()
    
    if rank == 0:
        print("=== Pain Points Analysis ===")
        print("Pain Point 1: User must remember exact initialization order")
        print("Pain Point 2: User must know which solver method to call")
        print("Pain Point 3: No clear workflow guidance")
        print("Pain Point 4: Error messages could be more user-friendly")
        print(f"Result: {theta_hat}")
        assert theta_hat.shape == (num_features,)
        print("✅ Pain points test completed")

def test_config_update_experience():
    """
    Test the experience of updating configuration.
    """
    num_agents = 30
    num_items = 8
    num_features = 2
    
    # Initial config
    cfg1 = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {"name": "Greedy"},
        "row_generation": {"max_iters": 10}
    }
    
    # Updated config (user wants to change some parameters)
    cfg2 = {
        "row_generation": {"max_iters": 20, "tolerance_optimality": 0.01}
    }
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        agent_features = np.random.normal(0, 1, (num_agents, num_features))
        obs_bundles = np.random.choice([0, 1], size=(num_agents, num_items), p=[0.5, 0.5])
        errors = np.random.normal(0, 0.1, size=(num_agents, num_items))
        
        input_data = {
            "agent_data": {"features": agent_features},
            "obs_bundle": obs_bundles,
            "errors": errors
        }
    else:
        input_data = None

    bc = BundleChoice()
    
    # Test config update experience
    bc.load_config(cfg1)
    bc.data.load_and_scatter(input_data)
    bc.features.set_oracle(simple_features_oracle)
    
    # User updates config
    bc.load_config(cfg2)
    
    # Test that components still work after config update
    theta_hat = bc.row_generation.solve()
    
    if rank == 0:
        print("=== Config Update Experience ===")
        print(f"Updated max_iters: {bc.config.row_generation.max_iters}")
        print(f"Updated tolerance_optimality: {bc.config.row_generation.tolerance_optimality}")
        print(f"Result: {theta_hat}")
        assert bc.config.row_generation.max_iters == 20
        assert bc.config.row_generation.tolerance_optimality == 0.01
        print("✅ Config update test passed") 