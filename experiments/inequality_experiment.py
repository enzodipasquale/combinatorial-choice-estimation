import os
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define dimensions
num_agents = 500
num_items = 100
num_features = 8
num_simuls = 1
sigma = 1

# Define configuration as a dictionary
cfg = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
    },
    "subproblem": {
        "name": "Greedy",
    }
}

# Load configuration
inequality_experiment = BundleChoice()
inequality_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
    errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
    estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    agent_data = {"modular": modular}
    data = {"agent_data": agent_data, 
            "errors": errors}
else:
    data = None

# Load and scatter data
inequality_experiment.data.load_and_scatter(data)

# Define features oracle
def features_oracle(i_id, bundle, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][i_id]

    if bundle.ndim == 1:
        return np.concatenate((modular_agent.T @ bundle, [-bundle.sum() ** 2]))
    else:
        return np.concatenate((modular_agent.T @ bundle, -np.sum(bundle, axis=0, keepdims=True) ** 2), axis=0)

inequality_experiment.features.set_oracle(features_oracle)
theta_0 = np.ones(num_features)
obs_bundles = inequality_experiment.subproblems.init_and_solve(theta_0)

# Estimate parameters using inequalities method
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
    print(f"aggregate: {obs_bundles.sum()}")
    data["obs_bundle"] = obs_bundles
    data["errors"] = estimation_errors
else:
    data = None

inequality_experiment.data.load_and_scatter(data)
inequality_experiment.features.set_oracle(features_oracle)
inequality_experiment.subproblems.load()

tic = datetime.now()
theta_hat = inequality_experiment.inequalities.solve()
elapsed = (datetime.now() - tic).total_seconds()

# Print estimation results
if rank == 0:
    print(f"estimation results: {theta_hat}")
    print(f"true theta: {theta_0}")
    print(f"elapsed time: {elapsed:.2f} seconds")
    
    # Calculate some statistics
    theta_diff = np.abs(theta_hat - theta_0)
    print(f"max absolute difference: {theta_diff.max():.6f}")
    print(f"mean absolute difference: {theta_diff.mean():.6f}")
    print(f"L2 norm difference: {np.linalg.norm(theta_diff):.6f}")

if rank == 0:
    # Validate inequalities at the estimated solution
    addOne = 0
    dropOne = 0
    DataArray = np.zeros((num_agents, num_items, num_features))
    
    for i in range(num_agents):
        features_i_obs = features_oracle(i, obs_bundles[i], data)
        for j in range(num_items):
            alt_bundle = obs_bundles[i].copy()
            if obs_bundles[i, j]:
                alt_bundle[j] = 0
                feat_alt_bundle = features_oracle(i, alt_bundle, data)
                dropOne += 1
            else:
                alt_bundle[j] = 1
                feat_alt_bundle = features_oracle(i, alt_bundle, data)
                addOne += 1 
            Delta_features = features_i_obs - feat_alt_bundle  
            DataArray[i, j, :] = Delta_features

    satisfied_at_star = ((DataArray @ theta_0) >= 0).sum()
    satisfied_at_hat = ((DataArray @ theta_hat) >= 0).sum()
    
    print(f"satisfied at true theta: {satisfied_at_star} out of {num_agents * num_items}")
    print(f"satisfied at estimated theta: {satisfied_at_hat} out of {num_agents * num_items}")
    print(f"addOne operations: {addOne}")
    print(f"dropOne operations: {dropOne}")
