import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/supermod"

# Define dimensions
num_agents = 100
num_items = 100
num_features = 6
num_simuls = 1

# Define configuration as a dictionary
modular_agent_features = 4
quadratic_item_features = 2
num_features = modular_agent_features + quadratic_item_features

cfg = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
    },
    "subproblem": {
        "name": "QuadSupermodularNetwork",
    },
    "rowgen": {
        "max_iters": 100,
        "tol_certificate": 0.001,
        "min_iters": 1,
        "master_settings": {
            "OutputFlag": 0
        }
    }
}

# Load configuration
quadsupermod_experiment = BundleChoice()
quadsupermod_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    modular_agent = - 10* np.random.normal(0, 1, (num_agents, num_items, modular_agent_features)) ** 2
    agent_data = {"modular": modular_agent}
    # quadratic_item = .5 * np.exp(-np.random.normal(0, 2, size=(num_items, num_items, quadratic_item_features)) ** 2)
    quadratic_item = np.random.choice([0, 1], size=(num_items, num_items, quadratic_item_features), p=[0.8, 0.2])
    for k in range(quadratic_item_features):
        np.fill_diagonal(quadratic_item[:, :, k], 0)
    item_data = {"quadratic": quadratic_item}
    errors = np.random.normal(0, 1, size=(num_agents, num_items)) * 10
    data = {"agent_data": agent_data, 
            "item_data": item_data, 
            "errors": errors}
else:
    data = None

quadsupermod_experiment.data.load_and_scatter(data)
quadsupermod_experiment.features.build_from_data()
beta_star = np.ones(num_features)
beta_star[-1:] = 0
obs_bundles = quadsupermod_experiment.subproblems.init_and_solve(beta_star)


# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
    data["obs_bundle"] = obs_bundles
    data["errors"] = np.random.normal(0, 1, size=(num_agents, num_items)) * 10
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, modular_agent_features)).to_csv(os.path.join(SAVE_PATH, "modular.csv"), index=False, header=False)
    pd.DataFrame(item_data["quadratic"].reshape(-1, quadratic_item_features)).to_csv(os.path.join(SAVE_PATH, "quadratic.csv"), index=False, header=False)
cfg["dimensions"]["num_simuls"] = num_simuls

quadsupermod_experiment.load_config(cfg)
quadsupermod_experiment.data.load_and_scatter(data)
quadsupermod_experiment.features.build_from_data()
quadsupermod_experiment.subproblems.load()
tic = datetime.now()
lambda_k_iter, p_j_iter = quadsupermod_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
obj_at_estimate = quadsupermod_experiment.row_generation.ObjVal(lambda_k_iter)
obj_at_star = quadsupermod_experiment.row_generation.ObjVal(beta_star)  

# Save estimation results as CSV
if rank == 0:
    print(f"estimation results:{lambda_k_iter}")
    print(beta_star)
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")
    row = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
        "subproblem": quadsupermod_experiment.subproblem_cfg.name,
        **{f"beta_star_{i}": val for i, val in enumerate(beta_star)},
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
    count = 0
    DataArray = np.zeros((num_agents, num_items, num_features))
    for i in range(num_agents):
        features_i_obs = quadsupermod_experiment.features.features_oracle(i, obs_bundles[i], data)
        for j in range(num_items):
            alt_bundle = obs_bundles[i].copy()
            if obs_bundles[i,j]:
                alt_bundle[j] = 0
                feat_alt_bundle = quadsupermod_experiment.features.features_oracle(i, alt_bundle, data)

            else:
                alt_bundle[j] = 1
                feat_alt_bundle = quadsupermod_experiment.features.features_oracle(i, alt_bundle, data)
  
            Delta_features = features_i_obs - feat_alt_bundle  
            count += 1
            DataArray[i,j,:] = Delta_features

    satisfied_at_star = ((DataArray @ beta_star) >= 0 ).sum()
    # print(f"satisfied_at_star: {satisfied_at_star} out of {num_agents * num_items}")