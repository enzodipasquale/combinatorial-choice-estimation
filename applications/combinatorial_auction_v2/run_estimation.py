#!/bin/env python
"""
Main estimation script for combinatorial auction v2.
"""

import sys
import os
import traceback
from pathlib import Path

# Add bundlechoice to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

def main():
    """Main execution with error handling."""
    try:
        # Load configuration
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            raise ValueError(f"Config file is empty or invalid: {CONFIG_PATH}")
        
        comm.Barrier()  # Sync after config load

        # Load data on rank 0
        if rank == 0:
            INPUT_DIR = os.path.join(BASE_DIR, "input_data")
            
            # Check required config keys
            required_keys = ["num_agents", "num_items", "num_features"]
            for key in required_keys:
                if key not in config.get("dimensions", {}):
                    raise KeyError(f"Missing config key: dimensions.{key}")
            
            num_agents = config["dimensions"]["num_agents"]
            num_items = config["dimensions"]["num_items"]
            num_features = config["dimensions"]["num_features"]
            # Backward compatibility: support both num_simulations and num_simuls
            num_simulations = config["dimensions"].get("num_simulations") or config["dimensions"].get("num_simuls", 1)
            
            # Check required files exist
            required_files = [
                "matching_i_j.npy",
                "quadratic_characteristic_j_j_k.npy",
                "weight_j.npy",
                "modular_characteristics_i_j_k.npy",
                "capacity_i.npy"
            ]
            for filename in required_files:
                filepath = os.path.join(INPUT_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Required file not found: {filepath}")
            
            obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))
            
            item_data = {
                "modular": -np.eye(num_items),
                "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
                "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
            }
            agent_data = {
                "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
                "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
            }
            
            # Validate dimensions
            if obs_bundle.shape != (num_agents, num_items):
                raise ValueError(f"obs_bundle shape mismatch: expected ({num_agents}, {num_items}), got {obs_bundle.shape}")
            
            np.random.seed(1995)
            errors = np.random.normal(0, 1, size=(num_simulations, num_agents, num_items))

            input_data = {
                "item_data": item_data,
                "agent_data": agent_data,
                "errors": errors,
                "obs_bundle": obs_bundle
            }
            
            print(f"Loaded data: {num_agents} agents, {num_items} items, {num_features} features, {num_simulations} simulations")
        else:
            input_data = None
            num_agents = None
            num_items = None
            num_features = None
            num_simulations = None

        comm.Barrier()  # Sync before broadcast
        num_agents = comm.bcast(num_agents, root=0)
        num_items = comm.bcast(num_items, root=0)
        num_features = comm.bcast(num_features, root=0)
        num_simulations = comm.bcast(num_simulations, root=0)

        # Run the estimation
        if rank == 0:
            print("\nInitializing BundleChoice...")
        
        combinatorial_auction = BundleChoice()
        combinatorial_auction.load_config(CONFIG_PATH)
        combinatorial_auction.data.load_and_scatter(input_data)
        combinatorial_auction.features.build_from_data()
        combinatorial_auction.subproblems.load()
        
        comm.Barrier()  # Sync after initialization
        
        if rank == 0:
            print("Starting row generation estimation...")
        
        result = combinatorial_auction.row_generation.solve()
        
        comm.Barrier()  # Sync after solve
        
        if rank == 0:
            print(f"\n{result.summary()}")
            
            # Save results
            OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            
            # Validate result
            if result.theta_hat is None:
                raise ValueError("Estimation failed: result.theta_hat is None")
            if len(result.theta_hat) != num_features:
                raise ValueError(f"Theta shape mismatch: expected {num_features}, got {len(result.theta_hat)}")
            
            # Save theta_hat
            theta_path = os.path.join(OUTPUT_DIR, "theta.npy")
            np.save(theta_path, result.theta_hat)
            
            # Save summary
            summary_path = os.path.join(OUTPUT_DIR, "theta_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(result.summary())
            
            print(f"\nResults saved to {OUTPUT_DIR}/")
            print(f"  Theta: {theta_path}")
            print(f"  Summary: {summary_path}")
            print("\n✓ DONE")
        
        comm.Barrier()  # Final sync
        
    except Exception as e:
        # Print error on all ranks for debugging
        error_msg = f"Rank {rank}: ERROR - {type(e).__name__}: {str(e)}"
        print(error_msg, file=sys.stderr, flush=True)
        if rank == 0:
            print("\n" + "=" * 60, file=sys.stderr)
            print("TRACEBACK (rank 0):", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        comm.Abort(1)  # Abort all ranks on error
        sys.exit(1)


if __name__ == "__main__":
    main()
