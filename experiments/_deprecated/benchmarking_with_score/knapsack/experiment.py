import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.factory import ScenarioLibrary
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/knapsack"


# Define dimensions
num_agents = 500
num_items = 100
num_simuls = 1
modular_agent_features = 3
modular_item_features = 2
num_features = modular_agent_features + modular_item_features
sigma = 1
seed = None  # No fixed seed for benchmarking

# Use factory to generate data (matches manual: abs(normal), weights 1-10, capacity with 0.5 mean_multiplier)
scenario = (
    ScenarioLibrary.linear_knapsack()
    .with_dimensions(num_agents=num_agents, num_items=num_items)
    .with_feature_counts(num_agent_features=modular_agent_features, num_item_features=modular_item_features)
    .with_num_simuls(num_simuls)
    .with_sigma(sigma)
    .with_capacity_config(mean_multiplier=0.5, lower_multiplier=0.85, upper_multiplier=1.15)  # Match manual
    .build()
)

# Use default theta_star (all ones)
theta_0 = np.ones(num_features)

# Prepare with theta_0 to avoid generating bundles twice
# This generates bundles internally and returns them in estimation_data
prepared = scenario.prepare(comm=comm, timeout_seconds=300, seed=seed, theta=theta_0)

# Get observed bundles from prepare() (already computed with theta_0) - only on rank 0
if rank == 0:
    obs_bundles = prepared.estimation_data["obs_bundle"]
    
    # Save data files
    agent_data = prepared.generation_data["agent_data"]
    item_data = prepared.generation_data["item_data"]
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, modular_agent_features)).to_csv(os.path.join(SAVE_PATH, "agent_modular.csv"), index=False, header=False)
    pd.DataFrame(item_data["modular"].reshape(-1, modular_item_features)).to_csv(os.path.join(SAVE_PATH, "item_modular.csv"), index=False, header=False)
    pd.DataFrame(agent_data["capacity"]).to_csv(os.path.join(SAVE_PATH, "capacity.csv"), index=False, header=False)
    pd.DataFrame(item_data["weights"]).to_csv(os.path.join(SAVE_PATH, "weights.csv"), index=False, header=False)

# Create BundleChoice for estimation (only when needed)
knapsack_experiment = BundleChoice()
prepared.apply(knapsack_experiment, comm=comm, stage="estimation")
knapsack_experiment.subproblems.load()
tic = datetime.now()
result = knapsack_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
theta_hat = result.theta_hat
obj_at_estimate = knapsack_experiment.row_generation.objective(theta_hat)
obj_at_star = knapsack_experiment.row_generation.objective(theta_0)  

# Save estimation results as CSV
if rank == 0:
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")

    weights = item_data["weights"]
    total_weight = (obs_bundles @ weights)
    # print( capacity - total_weight) 

    # print(f"{knapsack_experiment.row_generation.master_model.ObjVal}")
    row = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
        "subproblem": knapsack_experiment.subproblem_cfg.name,
        "sigma": sigma,
        **{f"theta_0_{i}": val for i, val in enumerate(theta_0)},
        **{f"beta_hat_{i}": val for i, val in enumerate(theta_hat)},
        "min_demand": obs_bundles.sum(1).min(),
        "max_demand": obs_bundles.sum(1).max(),
        "aggregate_demand": obs_bundles.sum(),
        "mean_demand": obs_bundles.sum(1).mean(),
        "elapsed": elapsed,
    }
    df = pd.DataFrame([row])
    output_path = os.path.join(BASE_DIR, "results.csv")
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False) 