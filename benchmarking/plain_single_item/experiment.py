import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/plain_single_item"

# Define dimensions
num_agents = 10000
num_items = 100
num_features = 5
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
        "name": "PlainSingleItem",
    },
    "row_generation": {
        "max_iters": 100,
        "tolerance_optimality": 0.001,
        "min_iters": 1,
        "gurobi_settings": {
            "OutputFlag": 0
        },
        # "max_slack_counter": 2,
    }
}

# Load configuration
plain_single_item_experiment = BundleChoice()
plain_single_item_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    # np.random.seed(1)
    modular_agent = np.random.normal(0, 1, (num_agents, num_items, num_features))
    while True:
        full_rank_matrix = np.random.randint(0,4, size=(num_features, num_features))
        if np.any(full_rank_matrix.sum(0) == 0):
            continue
        if np.linalg.matrix_rank(full_rank_matrix) == num_features:
            full_rank_matrix = (full_rank_matrix / full_rank_matrix.sum(0))
            break
    modular_agent =  modular_agent @ full_rank_matrix
    # modular_item = np.random.normal(0, 1, (num_items, 1))
    errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
    estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    agent_data = {"modular": modular_agent}
    # item_data = {"modular": modular_item}
    data = {"agent_data": agent_data, 
            # "item_data": item_data,
            "errors": errors}
else:
    data = None

# Load and scatter data
plain_single_item_experiment.data.load_and_scatter(data)
plain_single_item_experiment.features.build_from_data()
theta_0 = np.ones(num_features)
obs_bundles = plain_single_item_experiment.subproblems.init_and_solve(theta_0)

# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
    print(f"aggregate: {obs_bundles.sum()}")
    print(f"demand_j: {obs_bundles.sum(0)}")
    data["obs_bundle"] = obs_bundles
    data["errors"] = estimation_errors
    
    # Create directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, num_features-1)).to_csv(os.path.join(SAVE_PATH, "modular_agent.csv"), index=False, header=False)
    # pd.DataFrame(item_data["modular"]).to_csv(os.path.join(SAVE_PATH, "modular_item.csv"), index=False, header=False)
else:
    data = None

plain_single_item_experiment.load_config(cfg)
plain_single_item_experiment.data.load_and_scatter(data)
plain_single_item_experiment.features.build_from_data()
plain_single_item_experiment.subproblems.load()
tic = datetime.now()
lambda_k_iter = plain_single_item_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
obj_at_estimate = plain_single_item_experiment.row_generation.objective(lambda_k_iter)
obj_at_star = plain_single_item_experiment.row_generation.objective(theta_0)

# Save estimation results as CSV
if rank == 0:
    print(f"estimation results:{lambda_k_iter}")
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
        "subproblem": plain_single_item_experiment.subproblem_cfg.name,
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

# if rank == 0:
#     addOne = 0
#     dropOne = 0
#     DataArray = np.zeros((num_agents, num_items, num_features))
#     for i in range(num_agents):
#         features_i_obs = features_oracle(i, obs_bundles[i], data)
#         for j in range(num_items):
#             alt_bundle = obs_bundles[i].copy()
#             if obs_bundles[i,j]:
#                 alt_bundle[j] = 0
#                 feat_alt_bundle = features_oracle(i, alt_bundle, data)
#                 dropOne +=1
#             else:
#                 alt_bundle[j] = 1
#                 feat_alt_bundle = features_oracle(i, alt_bundle, data)
#                 addOne +=1 
#             Delta_features = features_i_obs - feat_alt_bundle  
#             DataArray[i,j,:] = Delta_features

#     satisfied_at_star = ((DataArray @ theta_0) >= 0 ).sum()
#     print(f"satisfied_at_star: {satisfied_at_star} out of {num_agents * num_items}") 