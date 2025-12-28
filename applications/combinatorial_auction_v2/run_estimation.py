#!/bin/env python

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI
import os
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    num_agents = config["dimensions"]["num_agents"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    num_simuls = config["dimensions"]["num_simuls"]

    item_data = {
        "modular": -np.eye(num_items),
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }
    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
    }

    np.random.seed(1995)
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    input_data = None

# Broadcast dimensions to all ranks
num_features = comm.bcast(num_features if rank == 0 else None, root=0)

# Run the estimation
combinatorial_auction = BundleChoice()
combinatorial_auction.load_config(CONFIG_PATH)
combinatorial_auction.data.load_and_scatter(input_data)
combinatorial_auction.features.build_from_data()
combinatorial_auction.subproblems.load()
theta = combinatorial_auction.row_generation.solve()

# Save results on rank 0
if rank == 0:
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate feature counts
    num_modular_agent = agent_data["modular"].shape[2]
    num_modular_item = item_data["modular"].shape[1]
    num_quadratic = item_data["quadratic"].shape[2]
    
    # Save theta as numpy array
    theta_path = os.path.join(OUTPUT_DIR, "theta.npy")
    np.save(theta_path, theta)
    print(f"\nSaved theta to {theta_path}")
    
    # Save theta with labels as CSV
    feature_labels = []
    feature_types = []
    
    # Modular agent features (indices 0 to num_modular_agent - 1)
    for k in range(num_modular_agent):
        feature_labels.append(f"modular_agent_{k}")
        feature_types.append("modular_agent")
    
    # Modular item features / fixed effects (indices num_modular_agent to num_modular_agent + num_modular_item - 1)
    for k in range(num_modular_item):
        feature_labels.append(f"modular_item_{k}")
        feature_types.append("modular_item")
    
    # Quadratic item features (indices num_modular_agent + num_modular_item to end)
    for k in range(num_quadratic):
        feature_labels.append(f"quadratic_item_{k}")
        feature_types.append("quadratic_item")
    
    df = pd.DataFrame({
        "feature_index": range(len(theta)),
        "feature_label": feature_labels,
        "feature_type": feature_types,
        "theta": theta
    })
    
    theta_csv_path = os.path.join(OUTPUT_DIR, "theta.csv")
    df.to_csv(theta_csv_path, index=False)
    print(f"Saved theta with labels to {theta_csv_path}")
    
    # Save summary statistics
    summary_path = os.path.join(OUTPUT_DIR, "theta_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Estimation Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features: {len(theta)}\n")
        f.write(f"  - Modular agent features: {num_modular_agent}\n")
        f.write(f"  - Modular item features (fixed effects): {num_modular_item}\n")
        f.write(f"  - Quadratic item features: {num_quadratic}\n\n")
        f.write(f"Theta statistics:\n")
        f.write(f"  Min: {theta.min():.6f}\n")
        f.write(f"  Max: {theta.max():.6f}\n")
        f.write(f"  Mean: {theta.mean():.6f}\n")
        f.write(f"  Std: {theta.std():.6f}\n")
    
    print(f"Saved summary to {summary_path}")






