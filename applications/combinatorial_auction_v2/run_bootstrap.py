#!/bin/env python
"""
Bootstrap subsampling script for combinatorial auction estimation.
"""

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI
import os
import pandas as pd
from pathlib import Path
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

NUM_REPLICATIONS = 1
BOOTSTRAP_WITH_REPLACEMENT = True
RANDOM_SEED = 1995

OUTPUT_DIR = os.path.join(BASE_DIR, "bootstrap_results")
if rank == 0:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load configuration (same as main estimation)
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0 (same structure as main estimation)
if rank == 0:
    print("=" * 70)
    if BOOTSTRAP_WITH_REPLACEMENT:
        print(f"Bootstrap: {NUM_REPLICATIONS} replication(s), ALL agents with replacement")
    else:
        print(f"Bootstrap: {NUM_REPLICATIONS} replications")
    print("=" * 70)
    
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    obs_bundle_full = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))
    
    num_agents_full = obs_bundle_full.shape[0]
    num_items_full = obs_bundle_full.shape[1]
    num_features = config["dimensions"]["num_features"]
    num_simuls = config["dimensions"]["num_simuls"]
    
    item_data_full = {
        "modular": -np.eye(num_items_full),
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }
    agent_data_full = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
    }
    
    np.random.seed(RANDOM_SEED)
    errors_full = np.random.normal(0, 1, size=(num_simuls, num_agents_full, num_items_full))
else:
    obs_bundle_full = None
    item_data_full = None
    agent_data_full = None
    errors_full = None
    num_agents_full = None
    num_items_full = None
    num_features = None
    num_simuls = None

# Broadcast dimensions to all ranks (same as main estimation)
num_features = comm.bcast(num_features if rank == 0 else None, root=0)
num_simuls = comm.bcast(num_simuls if rank == 0 else None, root=0)

# Bootstrap loop
all_results = []

for replication in range(NUM_REPLICATIONS):
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"REPLICATION {replication + 1}/{NUM_REPLICATIONS}")
        print(f"{'=' * 70}")
    
    # Bootstrap agents (with replacement)
    if rank == 0:
        np.random.seed(RANDOM_SEED + replication)
        if BOOTSTRAP_WITH_REPLACEMENT:
            agent_indices = np.random.choice(num_agents_full, size=num_agents_full, replace=True)
            num_agents_bootstrap = num_agents_full
        else:
            raise ValueError("Only bootstrap with replacement is supported")
        
        # Check MPI ranks vs agents
        num_mpi_ranks = comm.Get_size()
        if num_mpi_ranks > num_agents_bootstrap:
            print(f"ERROR: MPI ranks ({num_mpi_ranks}) > agents ({num_agents_bootstrap})")
            comm.Abort(1)
        
        # Get observed bundles for sampled agents (keep all items)
        obs_bundle_subsample = obs_bundle_full[agent_indices, :]
        
        # Keep all items (no filtering)
        num_items_subsample = num_items_full
        
        print(f"Agents: {num_agents_bootstrap}, Items: {num_items_subsample} (all items kept)")
        
        # Use all items (no filtering)
        item_data_subsample = {
            "modular": item_data_full["modular"].copy(),  # -np.eye(num_items_full)
            "quadratic": item_data_full["quadratic"].copy(),
            "weights": item_data_full["weights"].copy()
        }
        agent_data_subsample = {
            "modular": agent_data_full["modular"][agent_indices, :, :].copy(),
            "capacity": agent_data_full["capacity"][agent_indices].copy()
        }
        obs_bundle_subsample = obs_bundle_subsample.copy()  # Keep all items
        errors_subsample = errors_full[:, agent_indices, :].copy()
        
        # Calculate feature counts (same as full dataset)
        num_modular_agent = agent_data_subsample["modular"].shape[2]
        num_modular_item = item_data_subsample["modular"].shape[1]
        num_quadratic = item_data_subsample["quadratic"].shape[2]
        num_features_subsample = num_modular_agent + num_modular_item + num_quadratic
        
        # Create config for subsample (deep copy to avoid modifying original)
        config_subsample = copy.deepcopy(config)
        config_subsample["dimensions"]["num_agents"] = num_agents_bootstrap
        config_subsample["dimensions"]["num_items"] = num_items_subsample
        config_subsample["dimensions"]["num_features"] = num_features_subsample
        # Set upper bound to 5000 for bootstrap
        config_subsample["row_generation"]["theta_ubs"] = 5000
        # Set lower bounds: -1e9 for fixed effects only (indices 1 to num_modular_item)
        # Index 0 is modular agent, indices 1 to num_modular_item are item fixed effects
        theta_lbs = [0.0] * num_features_subsample  # Default: 0 for all
        for k in range(1, 1 + num_modular_item):  # Fixed effects: indices 1 to num_modular_item
            theta_lbs[k] = -1e9
        config_subsample["row_generation"]["theta_lbs"] = theta_lbs
        
        config_subsample_path = os.path.join(BASE_DIR, "config_bootstrap_temp.yaml")
        with open(config_subsample_path, 'w') as file:
            yaml.dump(config_subsample, file)
        
        input_data = {
            "item_data": item_data_subsample,
            "agent_data": agent_data_subsample,
            "errors": errors_subsample,
            "obs_bundle": obs_bundle_subsample
        }
        
        feature_info = {
            "num_modular_agent": num_modular_agent,
            "num_modular_item": num_modular_item,
            "num_quadratic": num_quadratic
        }
    else:
        input_data = None
        config_subsample_path = None
        feature_info = None
        num_items_subsample = None
        num_agents_bootstrap = None
    
    config_subsample_path = comm.bcast(config_subsample_path if rank == 0 else None, root=0)
    feature_info = comm.bcast(feature_info if rank == 0 else None, root=0)
    
    if rank == 0:
        print("Running estimation...")
    
    # Run estimation (same structure as main estimation)
    try:
        combinatorial_auction = BundleChoice()
        combinatorial_auction.load_config(config_subsample_path)
        combinatorial_auction.data.load_and_scatter(input_data)
        combinatorial_auction.features.build_from_data()
        combinatorial_auction.subproblems.load()
        theta = combinatorial_auction.row_generation.solve()
        
        if rank == 0:
            # Get bounds from config
            theta_ub = config_subsample["row_generation"].get("theta_ubs", 1000)
            theta_lb = config_subsample["row_generation"].get("theta_lbs", 0)
            if theta_lb is None:
                theta_lb = 0.0
            
            # Handle per-feature lower bounds (list) or scalar
            if isinstance(theta_lb, list):
                theta_lb_array = np.array(theta_lb)
                hitting_lb = np.where(np.abs(theta - theta_lb_array) < 1e-6)[0]
            else:
                theta_lb_array = np.full(len(theta), theta_lb)
                hitting_lb = np.where(np.abs(theta - theta_lb) < 1e-6)[0]
            
            # Check which features hit upper bound
            hitting_ub = np.where(np.abs(theta - theta_ub) < 1e-6)[0]
            
            num_modular_agent = feature_info["num_modular_agent"]
            num_modular_item = feature_info["num_modular_item"]
            num_quadratic = feature_info["num_quadratic"]
            
            print(f"\nBounds analysis:")
            if isinstance(theta_lb, list):
                print(f"  Lower bound: per-feature (0.0 for most, -1e9 for fixed effects), Upper bound: {theta_ub}")
            else:
                print(f"  Lower bound: {theta_lb}, Upper bound: {theta_ub}")
            if len(hitting_lb) > 0:
                print(f"  Features hitting LOWER bound: {hitting_lb.tolist()}")
            if len(hitting_ub) > 0:
                print(f"  Features hitting UPPER bound: {hitting_ub.tolist()}")
            
            # Classify which type of features are hitting bounds
            if len(hitting_lb) > 0 or len(hitting_ub) > 0:
                all_hitting = np.unique(np.concatenate([hitting_lb, hitting_ub]))
                modular_agent_hitting = all_hitting[all_hitting < num_modular_agent]
                modular_item_hitting = all_hitting[(all_hitting >= num_modular_agent) & (all_hitting < num_modular_agent + num_modular_item)]
                quadratic_hitting = all_hitting[all_hitting >= num_modular_agent + num_modular_item]
                
                print(f"\nBy feature type:")
                if len(modular_agent_hitting) > 0:
                    print(f"  Modular AGENT features hitting bounds: {modular_agent_hitting.tolist()}")
                if len(modular_item_hitting) > 0:
                    print(f"  Modular ITEM (fixed effects) hitting bounds: {modular_item_hitting.tolist()} ({len(modular_item_hitting)}/{num_modular_item})")
                if len(quadratic_hitting) > 0:
                    print(f"  Quadratic ITEM features hitting bounds: {quadratic_hitting.tolist()} ({len(quadratic_hitting)}/{num_quadratic})")
            
            result = {
                "replication": replication + 1,
                "num_agents": num_agents_bootstrap,
                "num_items": num_items_subsample,
                "num_modular_agent": num_modular_agent,
                "num_modular_item": num_modular_item,
                "num_quadratic": num_quadratic,
                "hitting_lb": hitting_lb.tolist() if len(hitting_lb) > 0 else [],
                "hitting_ub": hitting_ub.tolist() if len(hitting_ub) > 0 else [],
            }
            for i, val in enumerate(theta):
                result[f"theta_{i}"] = val
            all_results.append(result)
            
            # Clean up temp config
            if os.path.exists(config_subsample_path):
                os.remove(config_subsample_path)
    except Exception as e:
        if rank == 0:
            print(f"ERROR in replication {replication + 1}: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(config_subsample_path):
                os.remove(config_subsample_path)
        continue

# Save results
if rank == 0:
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(OUTPUT_DIR, "bootstrap_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n{'=' * 70}")
        print(f"Bootstrap complete! {len(all_results)}/{NUM_REPLICATIONS} successful")
        print(f"Results saved to: {csv_file}")
        print(f"{'=' * 70}")
    else:
        print(f"\nERROR: No successful replications!")
