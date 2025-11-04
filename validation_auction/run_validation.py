#!/bin/env python
"""
Run validation experiment: estimate parameters on synthetic data and compare with true parameters.

This script:
1. Loads synthetic data (bundles + theta_true)
2. Runs estimation on synthetic data
3. Compares theta_hat with theta_true
4. Saves results for analysis
"""

import numpy as np
import yaml
import os
import sys
import argparse
import json
import time
from pathlib import Path
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from bundlechoice import BundleChoice

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = Path(__file__).parent
AUCTION_DIR = BASE_DIR.parent / "applications" / "combinatorial_auction"

def compute_metrics(theta_true, theta_hat):
    """Compute validation metrics."""
    errors = theta_hat - theta_true
    metrics = {
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "mae": np.mean(np.abs(errors)),
        "bias": np.mean(errors),
        "bias_std": np.std(errors),
        "max_error": np.max(np.abs(errors)),
        "relative_error": np.mean(np.abs(errors) / (np.abs(theta_true) + 1e-10)) * 100,
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Run validation experiment')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory with synthetic data (default: synthetic_data/)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (default: config_small.yaml)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: results/)')
    parser.add_argument('--replication-id', type=int, default=0,
                       help='Replication ID for naming output')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else BASE_DIR / "synthetic_data"
    config_path = Path(args.config) if args.config else BASE_DIR / "config_small.yaml"
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "results"
    
    if rank == 0:
        print(f"Validation Experiment")
        print(f"  Data dir: {data_dir}")
        print(f"  Config: {config_path}")
        print(f"  Output dir: {output_dir}")
        
        # Check inputs
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            print("Please run generate_synthetic_data.py first")
            return 1
        
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return 1
        
        output_dir.mkdir(exist_ok=True)
    
    # Load synthetic data
    if rank == 0:
        obs_bundles = np.load(data_dir / "obs_bundles.npy")
        theta_true = np.load(data_dir / "theta_true.npy")
        
        # Load metadata
        if (data_dir / "metadata.json").exists():
            with open(data_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        print(f"\nLoaded synthetic data:")
        print(f"  Bundles shape: {obs_bundles.shape}")
        print(f"  Theta_true shape: {theta_true.shape}")
        
        num_agents = obs_bundles.shape[0]
        num_items = obs_bundles.shape[1]
        num_features = theta_true.shape[0]
    else:
        obs_bundles = None
        theta_true = None
        num_agents = None
        num_items = None
        num_features = None
    
    # Broadcast dimensions
    num_agents = comm.bcast(num_agents, root=0)
    num_items = comm.bcast(num_items, root=0)
    num_features = comm.bcast(num_features, root=0)
    
    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update config dimensions to match data
    config["dimensions"]["num_agents"] = num_agents
    config["dimensions"]["num_items"] = num_items
    config["dimensions"]["num_features"] = num_features
    
    # Load real data structure (for characteristics)
    if rank == 0:
        INPUT_DIR = AUCTION_DIR / "input_data"
        
        # Load full data
        full_modular_agent = np.load(INPUT_DIR / "modular_characteristics_i_j_k.npy")
        full_quadratic = np.load(INPUT_DIR / "quadratic_characteristic_j_j_k.npy")
        full_weights = np.load(INPUT_DIR / "weight_j.npy")
        full_capacity = np.load(INPUT_DIR / "capacity_i.npy")
        
        # Subsample if needed (use same indices as generation)
        if num_agents < full_modular_agent.shape[0]:
            agent_indices = np.arange(num_agents)  # Use first N agents
        else:
            agent_indices = np.arange(min(num_agents, full_modular_agent.shape[0]))
        
        if num_items < full_modular_agent.shape[1]:
            item_indices = np.arange(num_items)  # Use first N items
        else:
            item_indices = np.arange(min(num_items, full_modular_agent.shape[1]))
        
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
        
        # Generate errors for estimation
        num_simuls = config["dimensions"].get("num_simuls", 100)
        np.random.seed(42 + args.replication_id)
        errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
        
        input_data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": errors,
            "obs_bundle": obs_bundles,
        }
    else:
        input_data = None
    
    # Run estimation
    if rank == 0:
        print("\nRunning estimation on synthetic data...")
        start_time = time.time()
    
    combinatorial_auction = BundleChoice()
    combinatorial_auction.load_config(str(config_path))
    combinatorial_auction.data.load_and_scatter(input_data)
    combinatorial_auction.features.build_from_data()
    combinatorial_auction.subproblems.load()
    
    theta_hat = combinatorial_auction.row_generation.solve()
    
    if rank == 0:
        estimation_time = time.time() - start_time
        print(f"✓ Estimation completed in {estimation_time:.2f} seconds")
        
        # Compute metrics
        metrics = compute_metrics(theta_true, theta_hat)
        
        print(f"\nValidation Results:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Bias: {metrics['bias']:.6f} ± {metrics['bias_std']:.6f}")
        print(f"  Max Error: {metrics['max_error']:.6f}")
        print(f"  Relative Error: {metrics['relative_error']:.2f}%")
        
        # Save results
        results = {
            "theta_true": theta_true.tolist(),
            "theta_hat": theta_hat.tolist(),
            "metrics": metrics,
            "estimation_time": estimation_time,
            "replication_id": args.replication_id,
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
        }
        
        output_file = output_dir / f"results_{args.replication_id:03d}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as numpy for easy loading
        np.save(output_dir / f"theta_true_{args.replication_id:03d}.npy", theta_true)
        np.save(output_dir / f"theta_hat_{args.replication_id:03d}.npy", theta_hat)
        
        print(f"\n✓ Saved results to: {output_dir}")
        print(f"  Results file: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())

