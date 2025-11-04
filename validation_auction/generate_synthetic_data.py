#!/bin/env python
"""
Generate synthetic data using estimated parameters as true parameters.

This script:
1. Loads theta_hat from real data estimation (as theta_true)
2. Generates synthetic bundles using theta_true
3. Saves synthetic data for validation experiments
"""

import numpy as np
import yaml
import os
import sys
import argparse
from pathlib import Path
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = Path(__file__).parent
AUCTION_DIR = BASE_DIR.parent / "applications" / "combinatorial_auction"
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = BASE_DIR / "config_small.yaml"  # Use small config for generation

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--theta-file', type=str, default=None,
                       help='Path to theta_hat file (default: theta_hat_real.npy)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (default: config_small.yaml)')
    
    args = parser.parse_args()
    
    # Load theta_hat (as theta_true)
    if args.theta_file:
        theta_path = Path(args.theta_file)
    else:
        theta_path = BASE_DIR / "theta_hat_real.npy"
    
    if rank == 0:
        if not theta_path.exists():
            print(f"Error: Theta file not found: {theta_path}")
            print("Please run extract_parameters.py first")
            return 1
        
        theta_true = np.load(theta_path)
        print(f"Loaded theta_true from: {theta_path}")
        print(f"  Shape: {theta_true.shape}")
    else:
        theta_true = None
    
    # Load config first
    config_path = args.config or CONFIG_PATH
    if not config_path.exists():
        if rank == 0:
            print(f"Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update dimensions if using smaller config (on rank 0)
    if rank == 0:
        num_agents = config["dimensions"]["num_agents"]
        num_items = config["dimensions"]["num_items"]
        num_features_original = theta_true.shape[0]
        num_features_config = config["dimensions"].get("num_features", num_features_original)
        
        # For small config, we need to subsample theta_true to match
        # The feature structure is: modular_agent_features + items (identity) + quadratic_features
        # For small config, we'll use a subset of features
        if num_features_config < num_features_original:
            print(f"Warning: Config num_features ({num_features_config}) < theta_true size ({num_features_original})")
            print(f"Using first {num_features_config} features from theta_true")
            # Subsample theta_true to match config
            theta_true = theta_true[:num_features_config]
            num_features = num_features_config
        elif num_features_config > num_features_original:
            print(f"Warning: Config num_features ({num_features_config}) > theta_true size ({num_features_original})")
            print(f"Using theta_true size ({num_features_original}) for num_features")
            config["dimensions"]["num_features"] = num_features_original
            num_features = num_features_original
        else:
            config["dimensions"]["num_features"] = num_features_original
            num_features = num_features_original
    else:
        num_features = None
        num_agents = None
        num_items = None
    
    # Broadcast dimensions and theta_true (after potential subsampling)
    num_features = comm.bcast(num_features, root=0)
    num_agents = comm.bcast(num_agents, root=0)
    num_items = comm.bcast(num_items, root=0)
    
    if rank != 0:
        theta_true = np.empty(num_features, dtype=np.float64)
    theta_true = comm.bcast(theta_true, root=0)
    
    # Broadcast updated config
    config = comm.bcast(config if rank == 0 else None, root=0)
    
    if rank == 0:
        print(f"\nGenerating synthetic data:")
        print(f"  Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
        print(f"  Seed: {args.seed}")
    
    # Load real data structure (but we'll use smaller dimensions if config specifies)
    if rank == 0:
        INPUT_DIR = AUCTION_DIR / "input_data"
        
        # Load full data first
        full_obs_bundle = np.load(INPUT_DIR / "matching_i_j.npy")
        full_modular_agent = np.load(INPUT_DIR / "modular_characteristics_i_j_k.npy")
        full_quadratic = np.load(INPUT_DIR / "quadratic_characteristic_j_j_k.npy")
        full_weights = np.load(INPUT_DIR / "weight_j.npy")
        full_capacity = np.load(INPUT_DIR / "capacity_i.npy")
        
        num_agents = config["dimensions"]["num_agents"]
        num_items = config["dimensions"]["num_items"]
        
        # Subsample if needed
        if num_agents < full_obs_bundle.shape[0]:
            agent_indices = np.random.choice(full_obs_bundle.shape[0], num_agents, replace=False)
        else:
            agent_indices = np.arange(min(num_agents, full_obs_bundle.shape[0]))
        
        if num_items < full_obs_bundle.shape[1]:
            item_indices = np.random.choice(full_obs_bundle.shape[1], num_items, replace=False)
        else:
            item_indices = np.arange(min(num_items, full_obs_bundle.shape[1]))
        
        # Subsample data
        modular_agent = full_modular_agent[np.ix_(agent_indices, item_indices)]
        quadratic = full_quadratic[np.ix_(item_indices, item_indices)]
        weights = full_weights[item_indices]
        capacity = full_capacity[agent_indices]
        
        item_data = {
            "modular": -np.eye(num_items),
            "quadratic": quadratic,
            "weights": weights
        }
        agent_data = {
            "modular": modular_agent,
            "capacity": capacity,
        }
        
        # Generate errors for synthetic data (for single replication, shape is (num_agents, num_items))
        # Note: replications are handled at the validation stage, not generation
        np.random.seed(args.seed)
        errors = np.random.normal(0, 1, size=(num_agents, num_items))
        
        input_data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": errors,
        }
    else:
        input_data = None
    
    # Create output directory
    if rank == 0:
        output_dir = BASE_DIR / "synthetic_data"
        output_dir.mkdir(exist_ok=True)
    
    # Generate synthetic bundles for each replication
    combinatorial_auction = BundleChoice()
    combinatorial_auction.load_config(str(config_path))
    combinatorial_auction.data.load_and_scatter(input_data)
    combinatorial_auction.features.build_from_data()
    combinatorial_auction.subproblems.load()
    
    if rank == 0:
        print("\nGenerating synthetic bundles...")
    
    # Generate bundles using theta_true
    obs_bundles = combinatorial_auction.subproblems.init_and_solve(theta_true)
    
    if rank == 0:
        if obs_bundles is not None:
            print(f"✓ Generated synthetic bundles")
            print(f"  Shape: {obs_bundles.shape}")
            print(f"  Total demand: {obs_bundles.sum():.2f}")
            print(f"  Demand per agent: min={obs_bundles.sum(1).min():.2f}, max={obs_bundles.sum(1).max():.2f}")
            
            # Save synthetic data
            np.save(output_dir / "obs_bundles.npy", obs_bundles)
            np.save(output_dir / "theta_true.npy", theta_true)
            
            # Save metadata
            metadata = {
                "num_agents": num_agents,
                "num_items": num_items,
                "num_features": num_features,
                "seed": args.seed,
                "note": "Replications are handled at validation stage, not generation"
            }
            import json
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✓ Saved synthetic data to: {output_dir}")
        else:
            print("✗ Failed to generate bundles")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

