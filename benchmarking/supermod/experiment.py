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
num_agents = 1000
num_items = 100
num_simuls = 1
modular_agent_features = 5
quadratic_item_features = 1
num_features = modular_agent_features + quadratic_item_features
sigma = 5

cfg = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
    },
    "subproblem": {
        "name": "QuadSupermodularNetwork",
    },
    "row_generation": {
        "max_iters": 100,
        "min_iters": 1,
        "gurobi_settings": {
            "OutputFlag": 0
        },
        "theta_ubs": 100
    }
}

# Load configuration
quadsupermod_experiment = BundleChoice()
quadsupermod_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    # np.random.seed(64654534)

    # Modular agent features
    modular_agent = - 5* np.abs(np.random.normal(0, 1, (num_agents, num_items, modular_agent_features))) 
    # while True:
    #     full_rank_matrix = np.random.randint(0,3, size=(modular_agent_features, modular_agent_features))
    #     if np.any(full_rank_matrix.sum(0) == 0):
    #         continue
    #     if np.linalg.matrix_rank(full_rank_matrix) == modular_agent_features:
    #         full_rank_matrix = (full_rank_matrix / full_rank_matrix.sum(0))
    #         break
    # modular_agent = modular_agent @ full_rank_matrix
    agent_data = {"modular": modular_agent}

    # Quadratic item features
    # quadratic_item = .5 * np.exp(-np.random.normal(0, 2, size=(num_items, num_items, quadratic_item_features)) ** 2)
    quadratic_item = 1 * np.random.choice([0, 1], size=(num_items, num_items, quadratic_item_features), p=[0.8, 0.2])
    quadratic_item *= (1 - np.eye(num_items, dtype=int))[:,:, None]
    item_data = {"quadratic": quadratic_item}

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

quadsupermod_experiment.data.load_and_scatter(data)
quadsupermod_experiment.features.build_from_data()

theta_0 = np.ones(num_features)
obs_bundles, _ = quadsupermod_experiment.subproblems.init_and_solve(theta_0, return_values= True)

# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()},{obs_bundles.sum(1).mean()} , {obs_bundles.sum(1).max()}")
    print(f"demands: {obs_bundles.sum(1)}")
    data["obs_bundle"] = obs_bundles
    data["errors"] = estimation_errors
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, modular_agent_features)).to_csv(os.path.join(SAVE_PATH, "modular.csv"), index=False, header=False)
    pd.DataFrame(item_data["quadratic"].reshape(-1, quadratic_item_features)).to_csv(os.path.join(SAVE_PATH, "quadratic.csv"), index=False, header=False)

# Run row generation

cfg["dimensions"]["num_simuls"] = num_simuls
quadsupermod_experiment.load_config(cfg)
quadsupermod_experiment.data.load_and_scatter(data)
# quadsupermod_experiment.features.build_from_data()
# quadsupermod_experiment.subproblems.load()
# quadsupermod_experiment.subproblems.initialize_local()


tic = datetime.now()
theta_hat = quadsupermod_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
obj_at_estimate = quadsupermod_experiment.row_generation.objective(theta_hat)
obj_at_star = quadsupermod_experiment.row_generation.objective(theta_0)  

# Save estimation results as CSV
if rank == 0:
    print(theta_hat)
    print(theta_0)
    print(f"obj at estimate: {obj_at_estimate}")
    print(f"obj at star: {obj_at_star}")
    print(f"{quadsupermod_experiment.row_generation.master_model.ObjVal}")
    row = {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": num_simuls,
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