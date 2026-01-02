import sys
import os

# Add bundlechoice to path BEFORE importing bundlechoice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from bundlechoice.scenarios.data_generator import QuadraticGenerationMethod
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/supermod"


# Define dimensions
num_agents = 1000
num_items = 100
num_simulations = 1
modular_agent_features = 5
quadratic_item_features = 1
num_features = modular_agent_features + quadratic_item_features
sigma = 5
seed = None  # No fixed seed for benchmarking

# Use factory to generate data (matches manual: -5*abs(normal(0,1)), binary choice quadratic with p=0.2)
scenario = (
    ScenarioLibrary.quadratic_supermodular()
    .with_dimensions(num_agents=num_agents, num_items=num_items)
    .with_feature_counts(
        num_mod_agent=modular_agent_features,
        num_mod_item=0,
        num_quad_agent=0,
        num_quad_item=quadratic_item_features,
    )
    .with_num_simulations(num_simulations)
    .with_sigma(sigma)
    .with_agent_modular_config(multiplier=-5.0, mean=0.0, std=1.0)  # Match manual: -5*abs(normal) (apply_abs is always True)
    .with_quadratic_method(
        method=QuadraticGenerationMethod.BINARY_CHOICE,
        binary_prob=0.4,
        binary_value=1.0,
    )  # Match manual: binary choice with p=0.2
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
    pd.DataFrame(agent_data["modular"].reshape(-1, modular_agent_features)).to_csv(os.path.join(SAVE_PATH, "modular.csv"), index=False, header=False)
    pd.DataFrame(item_data["quadratic"].reshape(-1, quadratic_item_features)).to_csv(os.path.join(SAVE_PATH, "quadratic.csv"), index=False, header=False)

# Create BundleChoice for estimation (only when needed)
quadsupermod_experiment = BundleChoice()
prepared.apply(quadsupermod_experiment, comm=comm, stage="estimation")
quadsupermod_experiment.subproblems.load()


tic = datetime.now()
result = quadsupermod_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
theta_hat = result.theta_hat
obj_at_estimate = quadsupermod_experiment.row_generation.objective(theta_hat)
obj_at_star = quadsupermod_experiment.row_generation.objective(theta_0)  

# Save estimation results as CSV
if rank == 0:
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")
    row = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
        "subproblem": quadsupermod_experiment.subproblem_cfg.name,
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


# if rank == 0:
#     count = 0
#     DataArray = np.zeros((num_agents, num_items, num_features))
#     for i in range(num_agents):
#         features_i_obs = quadsupermod_experiment.features.features_oracle(i, obs_bundles[i], data)
#         for j in range(num_items):
#             alt_bundle = obs_bundles[i].copy()
#             if obs_bundles[i,j]:
#                 alt_bundle[j] = 0
#                 feat_alt_bundle = quadsupermod_experiment.features.features_oracle(i, alt_bundle, data)

#             else:
#                 alt_bundle[j] = 1
#                 feat_alt_bundle = quadsupermod_experiment.features.features_oracle(i, alt_bundle, data)
  
#             Delta_features = features_i_obs - feat_alt_bundle  
#             count += 1
#             DataArray[i,j,:] = Delta_features

#     satisfied_at_star = ((DataArray @ theta_0) >= 0 ).sum()
    # print(f"satisfied_at_star: {satisfied_at_star} out of {num_agents * num_items}") 