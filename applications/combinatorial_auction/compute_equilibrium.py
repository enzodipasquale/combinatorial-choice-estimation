#!/bin/env python
"""
Compute competitive equilibrium using estimated theta.

The equilibrium problem is:
    min_{p >= 0} sum_i u_i + sum_j p_j
    s.t. u_i >= sum_k feature_iBk * theta_hat_k - sum_{j in B} p_j + error_iB

We reformulate this for row generation:
- New "features" = -bundle (item indicators with -1), so feature @ p = -sum_{j in B} p_j
- New "error oracle" = original_features @ theta_hat + modular_error (without item FE)
"""

import sys
import os
import numpy as np
from mpi4py import MPI

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bundlechoice import BundleChoice
import yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Config - use smaller subset for debugging
DEBUG_MODE = True
NUM_AGENTS_DEBUG = 40  # Use subset of agents
NUM_ITEMS_DEBUG = 10   # Small enough for brute force (2^10 = 1024 bundles)

IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    
    # Load original data
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))
    modular_chars = np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy"))
    quadratic_chars = np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy"))
    capacity = np.load(os.path.join(INPUT_DIR, "capacity_i.npy"))
    weights = np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    
    # Load or estimate theta_hat
    THETA_PATH = os.path.join(BASE_DIR, "estimation_results", "theta.npy")
    if os.path.exists(THETA_PATH):
        theta_hat = np.load(THETA_PATH)
        print(f"Loaded theta_hat from {THETA_PATH}")
    else:
        # Use dummy theta for testing (should run estimation first)
        num_items_full = obs_bundle.shape[1]
        theta_hat = np.ones(num_items_full + 4)  # item FE + 1 agent mod + 3 quad
        print("WARNING: Using dummy theta_hat. Run estimation first for real values.")
    
    # For debugging, use subset - pick agents with actual bundles
    if DEBUG_MODE:
        # Find agents with largest bundles
        agent_bundle_sizes = obs_bundle.sum(axis=1)
        top_agent_ids = np.argsort(agent_bundle_sizes)[-NUM_AGENTS_DEBUG:]
        
        # Get items that these agents actually want
        selected_bundles = obs_bundle[top_agent_ids]
        item_demand = selected_bundles.sum(axis=0)
        top_item_ids = np.argsort(item_demand)[-NUM_ITEMS_DEBUG:]
        top_item_ids = np.sort(top_item_ids)  # Keep sorted for consistency
        
        # Subset data
        num_agents = len(top_agent_ids)
        num_items = len(top_item_ids)
        obs_bundle = obs_bundle[np.ix_(top_agent_ids, top_item_ids)]
        modular_chars = modular_chars[np.ix_(top_agent_ids, top_item_ids, np.arange(modular_chars.shape[2]))]
        quadratic_chars = quadratic_chars[np.ix_(top_item_ids, top_item_ids, np.arange(quadratic_chars.shape[2]))]
        capacity = capacity[top_agent_ids]
        weights = weights[top_item_ids]
        
        # Subset theta_hat: selected item FE + last 4 (agent mod + quad)
        # Original structure: 493 item FE + 1 agent modular + 3 item quadratic = 497
        theta_hat = np.concatenate([theta_hat[top_item_ids], theta_hat[-4:]])
        
        print(f"Selected {num_agents} agents with bundle sizes: {obs_bundle.sum(axis=1)}")
        print(f"Selected {num_items} items")
    else:
        num_agents = obs_bundle.shape[0]
        num_items = obs_bundle.shape[1]
    
    print(f"Equilibrium computation: {num_agents} agents, {num_items} items")
    print(f"Theta hat shape: {theta_hat.shape}")
    
    # Generate errors (same as estimation for reproducibility)
    np.random.seed(1995)
    num_simulations = config["dimensions"]["num_simulations"]
    all_errors = np.random.normal(0, 1, size=(num_simulations, config["dimensions"]["num_agents"], num_items))
    # Take just first simulation for equilibrium computation
    if DEBUG_MODE:
        modular_errors = all_errors[0, top_agent_ids, :]
    else:
        modular_errors = all_errors[0, :num_agents, :]
    
    # item_data modular = -I (gives -sum_{j in B} p_j when multiplied by p)
    item_modular_for_prices = -np.eye(num_items)
    
    input_data = {
        "item_data": {
            "modular": item_modular_for_prices,  # -I for price features only
        },
        "agent_data": {},  # No agent features for equilibrium prices
        "errors": modular_errors,
        "obs_bundle": obs_bundle,
    }
    
    # Prepare data for error oracle (to be broadcast)
    oracle_data = {
        "modular_chars": modular_chars,
        "quadratic_chars": quadratic_chars,
        "modular_errors": modular_errors,
    }
else:
    input_data = None
    num_agents = None
    num_items = None
    theta_hat = None
    oracle_data = None

# Broadcast all necessary data
num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
theta_hat = comm.bcast(theta_hat, root=0)
oracle_data = comm.bcast(oracle_data, root=0)

# Create the equilibrium BundleChoice
bc = BundleChoice()
bc.load_config({
    'dimensions': {
        'num_agents': num_agents,
        'num_items': num_items,
        'num_features': num_items,  # prices are the "features" now
    },
    # Use BruteForce solver - works with custom error oracles (only for small J)
    'subproblem': {'name': 'BruteForce'},
    'row_generation': {
        'max_iters': 100,
        'theta_ubs': [1000] * num_items,  # Upper bound on prices
        'theta_lbs': [0] * num_items,     # Prices >= 0
    },
})

bc.data.load_and_scatter(input_data)

# Build features oracle from data (for -bundle @ p part)
bc.features.build_features_oracle_from_data()


def make_equilibrium_error_oracle(theta_hat, oracle_data, global_start_idx, num_items):
    """Create error oracle that includes deterministic utility WITHOUT item fixed effects.
    
    In the original estimation:
      U_iB = agent_modular @ theta_1 + (-I) @ theta_item_fe + quadratic @ theta_quad + error
    
    The item FE represent "prices" in the estimation. For equilibrium, we replace 
    them with the new price variables p_j. So the error oracle computes:
      utility_iB = agent_modular @ theta_agent + quadratic @ theta_quad + modular_error
    
    The prices enter through the features (-I @ p).
    """
    modular_chars = oracle_data["modular_chars"]
    quadratic_chars = oracle_data["quadratic_chars"]
    modular_errors = oracle_data["modular_errors"]
    
    # Extract theta components:
    # theta_hat = [theta_item_fe (num_items), theta_agent_mod (1), theta_item_quad (3)]
    theta_agent_mod = theta_hat[num_items:num_items+1]  # 1 feature
    theta_item_quad = theta_hat[num_items+1:]           # 3 features
    
    def equilibrium_error_oracle(local_id, bundle, data):
        # Convert local_id to global agent id
        global_id = global_start_idx + local_id
        
        # Get modular error
        modular_err = (modular_errors[global_id] * bundle).sum()
        
        # Compute utility WITHOUT item FE (they become price variables)
        # Agent modular features
        agent_modular_i = modular_chars[global_id]
        agent_feat = np.einsum('jk,j->k', agent_modular_i, bundle)
        agent_utility = agent_feat @ theta_agent_mod
        
        # Item quadratic 
        quad_feat = np.einsum('jlk,j,l->k', quadratic_chars, bundle, bundle)
        quad_utility = quad_feat @ theta_item_quad
        
        total_utility = agent_utility + quad_utility + modular_err
        
        return float(total_utility)
    
    return equilibrium_error_oracle


# Compute global start index for this rank
size = comm.Get_size()
agents_per_rank = num_agents // size
remainder = num_agents % size
global_start_idx = rank * agents_per_rank + min(rank, remainder)

# Set the custom error oracle
bc.features.set_error_oracle(make_equilibrium_error_oracle(theta_hat, oracle_data, global_start_idx, num_items))

bc.subproblems.load()
bc.subproblems.initialize_local()

if rank == 0:
    print("Starting equilibrium row generation...")
    print(f"Price bounds: [0, 1000]")

# Solve for equilibrium prices
result = bc.row_generation.solve()

if rank == 0:
    print(f"\n{result.summary()}")
    
    if result.converged:
        prices = result.theta_hat
        print(f"\nEquilibrium prices: min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f}")
        
        # Save results
        from pathlib import Path
        OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(OUTPUT_DIR, "equilibrium_prices.npy"), prices)
        print(f"Saved equilibrium prices to {OUTPUT_DIR}/equilibrium_prices.npy")
