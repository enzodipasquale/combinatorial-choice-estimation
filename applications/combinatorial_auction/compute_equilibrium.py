#!/bin/env python
"""
Compute competitive equilibrium using estimated theta.

The equilibrium problem is:
    min_{p >= 0} sum_i u_i + sum_j p_j
    s.t. u_i >= sum_k feature_iBk * theta_hat_k - sum_{j in B}p_j + error_iB

Strategy: Use full feature structure but fix non-price theta components.
- theta = [p_1, ..., p_J, theta_agent_mod (fixed), theta_item_quad (fixed)]
- Prices (first num_items components) are free in [0, max_price]
- Other components are fixed to theta_hat values via bounds
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import csv

BASE_DIR = os.path.dirname(__file__)  # combinatorial_auction/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))  # combinatorial-choice-estimation/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Config
NUM_AGENTS_SUBSET = 40  # Restrict to 40 agents for faster computation
NUM_SIMULATIONS = 1     # Use 1 simulation

IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "point_estimate", "config_local.yaml" if IS_LOCAL else "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "data", "114402-V1", "input_data", "delta4")
    obs_bundle_full = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))
    
    num_obs_full = config["dimensions"]["num_obs"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    
    # Subset to top agents by bundle size
    agent_bundle_sizes = obs_bundle_full.sum(axis=1)
    top_agent_ids = np.argsort(agent_bundle_sizes)[-NUM_AGENTS_SUBSET:]
    
    num_obs = len(top_agent_ids)
    obs_bundle = obs_bundle_full[top_agent_ids]
    
    print(f"Selected {num_obs} agents (top by bundle size)")
    print(f"Bundle sizes: min={obs_bundle.sum(axis=1).min()}, max={obs_bundle.sum(axis=1).max()}, mean={obs_bundle.sum(axis=1).mean():.1f}")
    
    # Load feature data (subset agents)
    item_data = {
        "modular": -np.eye(num_items),  # Item fixed effects (will become prices)
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }
    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy"))[top_agent_ids],
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy"))[top_agent_ids],
    }
    
    # Generate errors (1 simulation only)
    np.random.seed(1995)
    errors_full = np.random.normal(0, 1, size=(NUM_SIMULATIONS, num_obs_full, num_items))
    errors = errors_full[:, top_agent_ids, :]  # Shape: (1, num_obs, num_items)
    
    # Load estimated theta
    THETA_PATH = os.path.join(BASE_DIR, "estimation_results", "theta.npy")
    if os.path.exists(THETA_PATH):
        theta_hat = np.load(THETA_PATH)
        print(f"Loaded theta_hat from {THETA_PATH}")
    else:
        raise FileNotFoundError(f"theta_hat not found at {THETA_PATH}. Run estimation first.")
    
    print(f"theta_hat shape: {theta_hat.shape}")
    print(f"First 5 (item FE): {theta_hat[:5]}")
    print(f"Last 4 (agent_mod + quad): {theta_hat[-4:]}")
    
    input_data = {
        "item_data": item_data,
        "obs_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    input_data = None
    num_obs = None
    num_items = None
    num_features = None
    theta_hat = None

# Broadcast dimensions to all ranks
num_obs = comm.bcast(num_obs, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
theta_hat = comm.bcast(theta_hat, root=0)

# For equilibrium:
# - First num_items theta components are prices (free, >= 0)
# - Last 4 theta components are fixed to theta_hat values (agent_mod=1, item_quad=3)
MAX_PRICE = 1000.0

# Build bounds: prices in [0, MAX_PRICE], other components fixed to theta_hat
theta_lbs = [0.0] * num_items + list(theta_hat[-4:])
theta_ubs = [MAX_PRICE] * num_items + list(theta_hat[-4:])

if rank == 0:
    print(f"\nEquilibrium setup:")
    print(f"  Agents: {num_obs}, Items: {num_items}, Features: {num_features}")
    print(f"  Price variables: {num_items} (bounds [0, {MAX_PRICE}])")
    print(f"  Fixed theta components: {theta_hat[-4:]}")

# Run the equilibrium computation
bc = BundleChoice()
bc.load_config({
    'dimensions': {
        'num_obs': num_obs,
        'num_items': num_items,
        'num_features': num_features,
        'num_simulations': NUM_SIMULATIONS,
    },
    'subproblem': config['subproblem'],  # Use same subproblem as estimation (QuadKnapsack)
    'row_generation': {
        'max_iters': 200,
        'theta_lbs': theta_lbs,
        'theta_ubs': theta_ubs,
    },
})

bc.data.load_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.subproblems.load()

if rank == 0:
    print("\nStarting equilibrium row generation...")

result = bc.row_generation.solve()

if rank == 0:
    print(f"\n{result.summary()}")
    
    if result.converged:
        # Extract prices (first num_items components)
        prices = result.theta_hat[:num_items]
        fixed_theta = result.theta_hat[num_items:]
        
        print(f"\n=== EQUILIBRIUM RESULTS ===")
        print(f"Prices: min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f}")
        print(f"Fixed theta (should match theta_hat): {fixed_theta}")
        print(f"Theta_hat[-4:] for comparison: {theta_hat[-4:]}")
        
        # Verify fixed components didn't change
        if np.allclose(fixed_theta, theta_hat[-4:], atol=1e-6):
            print("✅ Fixed theta components match theta_hat")
        else:
            print("⚠️ Fixed theta components differ from theta_hat")
        
        # Save results
        OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "equilibrium_prices.npy"), prices)
        print(f"\nSaved equilibrium prices to {OUTPUT_DIR}/equilibrium_prices.npy")
