import os
import sys

# Add bundlechoice to path BEFORE importing bundlechoice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from datetime import datetime


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/greedy"

# Define dimensions
num_agents = 1000
num_items = 100
num_features = 5
num_simulations = 1
sigma = 1
seed = None  # No fixed seed for benchmarking

# Use factory to generate data (matches manual: abs(normal(0,1)))
scenario = (
    ScenarioLibrary.greedy()
    .with_dimensions(num_agents=num_agents, num_items=num_items)
    .with_num_features(num_features)
    .with_num_simulations(num_simulations)
    .with_sigma(sigma)
    .build()
)

# Create custom theta
theta_0 = np.ones(num_features)
theta_0[-1] = 0.1  # Custom theta for this experiment

# Prepare with custom theta to avoid generating bundles twice
# This generates bundles internally and returns them in estimation_data
prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed, theta=theta_0)

# Get observed bundles from prepare() (already computed with theta_0) - only on rank 0
if rank == 0:
    obs_bundles = prepared.estimation_data["obs_bundle"]
    # Save data files
    agent_data = prepared.generation_data["agent_data"]
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, num_features-1)).to_csv(os.path.join(SAVE_PATH, "modular.csv"), index=False, header=False)

# Create BundleChoice for estimation (only when needed)
greedy_experiment = BundleChoice()
prepared.apply(greedy_experiment, comm=comm, stage="estimation")
greedy_experiment.subproblems.load()
tic = datetime.now()
result = greedy_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
lambda_k_iter = result.theta_hat
obj_at_estimate = greedy_experiment.row_generation.objective(lambda_k_iter)
obj_at_star = greedy_experiment.row_generation.objective(theta_0)
# Save estimation results as CSV
if rank == 0:
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "subproblem": greedy_experiment.subproblem_cfg.name,
        "sigma": sigma,
        **{f"theta_0_{i}": val for i, val in enumerate(theta_0)},
        **{f"beta_hat_{i}": val for i, val in enumerate(lambda_k_iter)},
        "min_demand": obs_bundles.sum(1).min(),
        "max_demand": obs_bundles.sum(1).max(),
        "aggregate_demand": obs_bundles.sum(),
        "mean_demand": obs_bundles.sum(1).mean(),
        "elapsed": elapsed,
    }
    df = pd.DataFrame([row])
    output_path = os.path.join(BASE_DIR, "results.csv")
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False) 



if rank == 0:
    addOne = 0
    dropOne = 0
    DataArray = np.zeros((num_agents, num_items, num_features))
    data = prepared.estimation_data
    for i in range(num_agents):
        features_i_obs = greedy_experiment.oracles.features_oracle(i, obs_bundles[i], data)
        for j in range(num_items):
            alt_bundle = obs_bundles[i].copy()
            if obs_bundles[i,j]:
                alt_bundle[j] = 0
                feat_alt_bundle = greedy_experiment.oracles.features_oracle(i, alt_bundle, data)
                dropOne +=1

            else:
                alt_bundle[j] = 1
                feat_alt_bundle = greedy_experiment.oracles.features_oracle(i, alt_bundle, data)
                addOne +=1 
            Delta_features = features_i_obs - feat_alt_bundle  
            DataArray[i,j,:] = Delta_features

    satisfied_at_star = ((DataArray @ theta_0) >= 0 ).sum()
    print(f"satisfied_at_star: {satisfied_at_star} out of {num_agents * num_items}")


