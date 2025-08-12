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
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/knapsack"


# Define dimensions
num_agents = 200
num_items = 100
num_simuls = 1
modular_agent_features = 4
modular_item_features = 2
num_features = modular_agent_features + modular_item_features
sigma = 1

cfg = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
    },
    "subproblem": {
        "name": "LinearKnapsack",
        "settings": {
            "TimeLimit": 60,
            "MIPGap_tol": 0.01
        }
    },
    "rowgen": {
        "max_iters": 100,
        "tolerance_optimality": 0.001,
        "master_settings": {
            "OutputFlag": 0
        }
    }
}

# Load configuration
knapsack_experiment = BundleChoice()
knapsack_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    # np.random.seed(64654534)

    # Modular agent features
    modular_agent = np.abs(np.random.normal(0, 1, (num_agents, num_items, modular_agent_features)) )
    agent_data = {"modular": modular_agent}

    # Modular item features (weights)
    modular_item = np.abs(np.random.normal(0, 1, (num_items, modular_item_features)))
    weights = np.random.randint(0, 10, num_items)
    item_data = {"modular": modular_item,
                 "weights": weights}

    # Agent capacities
    capacity = np.random.randint(5, 30, num_agents)
    agent_data["capacity"] = capacity

    # Errors
    errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
    estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

    # Data
    data = {"agent_data": agent_data, 
            "item_data": item_data, 
            "errors": errors
            }
else:
    data = None

knapsack_experiment.data.load_and_scatter(data)
knapsack_experiment.features.build_from_data()

theta_0 = np.ones(num_features)
obs_bundles, _ = knapsack_experiment.subproblems.init_and_solve(theta_0, return_values= True)
        
# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()},{obs_bundles.sum(1).mean()} , {obs_bundles.sum(1).max()}")
    
    data["obs_bundle"] = obs_bundles
    data["errors"] = estimation_errors
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, modular_agent_features)).to_csv(os.path.join(SAVE_PATH, "agent_modular.csv"), index=False, header=False)
    pd.DataFrame(item_data["modular"].reshape(-1, modular_item_features)).to_csv(os.path.join(SAVE_PATH, "item_modular.csv"), index=False, header=False)
    pd.DataFrame(capacity).to_csv(os.path.join(SAVE_PATH, "capacity.csv"), index=False, header=False)
    pd.DataFrame(weights).to_csv(os.path.join(SAVE_PATH, "weights.csv"), index=False, header=False)

# Run row generation
cfg["dimensions"]["num_simuls"] = num_simuls
knapsack_experiment.load_config(cfg)
knapsack_experiment.data.load_and_scatter(data)
knapsack_experiment.features.build_from_data()
knapsack_experiment.subproblems.load()
tic = datetime.now()
theta_hat = knapsack_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
obj_at_estimate = knapsack_experiment.row_generation.objective(theta_hat)
obj_at_star = knapsack_experiment.row_generation.objective(theta_0)  

# Save estimation results as CSV
if rank == 0:
    print(theta_hat)
    print(theta_0)
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")

    total_weight = (obs_bundles @ weights)
    # print( capacity - total_weight) 

    # print(f"{knapsack_experiment.row_generation.master_model.ObjVal}")
    row = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
        "subproblem": knapsack_experiment.subproblem_cfg.name,
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