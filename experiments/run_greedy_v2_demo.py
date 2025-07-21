#!/usr/bin/env python3
"""
V2 Demo: Greedy Subproblem with Row Generation
This demo shows how to use the new v2 API with synthetic data and row generation solver.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bundlechoice.v2.core import BundleChoice
from bundlechoice.v2.config import DimensionsConfig, SubproblemConfig
from bundlechoice.v2.data_manager import DataManager
from bundlechoice.v2.feature_manager import FeatureManager
from bundlechoice.v2.subproblems.manager import SubproblemManager

def generate_synthetic_data(num_agents=10, num_items=8, agent_modular_dim=2, item_modular_dim=3, num_simuls=2):
    """Generate synthetic data for the demo, matching the integration test structure."""
    np.random.seed(42)
    item_features = np.random.normal(0, 1, (num_items, item_modular_dim))
    agent_features = np.random.normal(0, 1, (num_agents, num_items, agent_modular_dim))
    # Do not generate errors here; generate separately for generation and estimation
    return {
        'item_features': item_features,
        'agent_features': agent_features
    }

def get_features(i_id, B_j, data):
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_item = data["item_data"]["modular"]
    return np.concatenate((modular_agent[B_j].sum(0),
                           modular_item[B_j].sum(0),
                           [-B_j.sum() ** 2]))

def main():
    print("=== V2 Greedy Demo with Row Generation ===")
    
    # Configuration for synthetic data
    num_agents = 100
    num_items = 6
    agent_modular_dim = 2
    item_modular_dim = 3
    num_features = agent_modular_dim + item_modular_dim + 1
    num_simuls = 2  # For estimation
    
    print(f"Configuration: {num_agents} agents, {num_items} items, {num_features} features, {num_simuls} simulations")
    
    # Generate synthetic data (no obs_bundles yet)
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(num_agents, num_items, agent_modular_dim, item_modular_dim, num_simuls)
    
    # Set true parameters to all ones
    true_params = np.ones(num_features)
    
    # --- Step 1: Generate obs_bundles using true_params and num_simuls=1 ---
    print("2. Generating observed bundles using true parameters...")
    # Generate i.i.d. N(0,1) errors for bundle generation
    errors_gen = np.random.normal(0, 1, (1, num_agents, num_items))
    cfg_gen = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1
        },
        "subproblem": {
            "name": "Greedy",
            "settings": {
                "parallel_local": True,
                "multithreading": False
            }
        }
    }
    bc_gen = BundleChoice()
    bc_gen.load_config(cfg_gen)
    input_data_gen = {
        "item_data": {"modular": data['item_features']},
        "agent_data": {"modular": data['agent_features']},
        "errors": errors_gen.reshape(1 * num_agents, num_items),
    }
    bc_gen.load_data(input_data_gen, scatter=True)
    bc_gen.load_features(get_features)
    bc_gen._try_init_subproblem_manager()
    obs_bundles_raw = bc_gen.init_and_solve_subproblems(true_params)
    if bc_gen.rank == 0 and obs_bundles_raw is not None:
        obs_bundles = obs_bundles_raw.reshape(num_agents, num_items)
        print(f"Generated {num_agents} observed bundles with shape {obs_bundles.shape}")
    else:
        obs_bundles = None
    obs_bundles = bc_gen.comm.bcast(obs_bundles, root=0)
    if bc_gen.rank == 0:
        print(f"True parameters: {true_params}")
        print(f"Sample bundles (first 3 agents):")
        for i in range(min(3, num_agents)):
            bundle = obs_bundles[i]
            if bc_gen.get_features is not None:
                features = bc_gen.get_features(i, bundle, bc_gen.input_data)
                print(f"  Agent {i}: bundle={bundle}, features={features}")
            else:
                print(f"  Agent {i}: bundle={bundle}")
    
    # --- Step 2: Verify bundle generation results ---
    print("3. Verifying bundle generation results...")
    if bc_gen.rank == 0:
        print(f"Observed bundles shape: {obs_bundles.shape}")
        print(f"Expected shape: ({num_agents}, {num_items})")
        print(f"Bundle values are binary: {np.all(np.logical_or(obs_bundles == 0, obs_bundles == 1))}")
        print(f"Number of bundles with size > 1: {np.sum(np.sum(obs_bundles, axis=1) > 1)}")
        print(f"Average bundle size: {np.mean(np.sum(obs_bundles, axis=1)):.2f}")
        print(f"Bundle size distribution: {np.bincount(np.sum(obs_bundles, axis=1).astype(int))}")
        
        # Show some sample bundles
        print("\nSample bundles (first 5 agents):")
        for i in range(min(5, num_agents)):
            bundle = obs_bundles[i]
            bundle_size = np.sum(bundle)
            selected_items = np.where(bundle)[0]
            print(f"  Agent {i}: bundle={bundle}, size={bundle_size}, items={selected_items}")
    
    print("\nBundle generation completed successfully!")
    
    # Comment out estimation for now
    """
    # --- Step 2: Set up estimation config and data ---
    print("3. Setting up estimation configuration...")
    # Generate i.i.d. N(0,1) errors for estimation
    errors_est = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": num_simuls
        },
        "subproblem": {
            "name": "Greedy",
            "settings": {
                "parallel_local": True,
                "multithreading": False
            }
        },
        "rowgen": {
            "max_iters": 10,
            "tol_certificate": 1e-2
        }
    }
    input_data = {
        "item_data": {"modular": data['item_features']},
        "agent_data": {"modular": data['agent_features']},
        "errors": errors_est.reshape(num_simuls * num_agents, num_items),
        "obs_bundle": obs_bundles
    }
    print("4. Initializing BundleChoice for estimation...")
    bc = BundleChoice()
    bc.load_config(cfg)
    bc.load_data(input_data, scatter=True)
    bc.load_features(get_features)
    bc._try_init_subproblem_manager()
    print("5. Running row generation solver...")
    from bundlechoice.v2.compute_estimator.row_generation import RowGenerationSolver
    solver = RowGenerationSolver(bc, bc.rowgen_cfg)
    lambda_k, p_j = solver.compute_estimator_row_gen()
    print("\n=== Results ===")
    if bc.rank == 0:
        print(f"Estimated parameters (lambda_k): {lambda_k}")
        if p_j is not None:
            print(f"Estimated prices (p_j): {p_j}")
        print(f"True parameters: {true_params}")
        if lambda_k is not None:
            mse = np.mean((lambda_k - true_params)**2)
            print(f"Mean squared error: {mse:.6f}")
            if mse < 0.5:
                print("TEST PASSED: Row generation recovers parameters closely.")
            else:
                print("TEST FAILED: Row generation did not recover parameters closely enough.")
            assert mse < 0.5, f"Row generation MSE too high: {mse}"
    print("\nDemo completed successfully!")
    """

if __name__ == "__main__":
    main() 