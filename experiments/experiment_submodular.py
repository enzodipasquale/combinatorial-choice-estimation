import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
SAVE_PATH = "/Users/enzo-macbookpro/MyProjects/score-estimator/greedy"

# Define dimensions
num_agents = 1500
num_items = 100
num_features = 5
num_simuls = 1
sigma = 0

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
    },
    "rowgen": {
        "max_iters": 100,
        "tolerance_optimality": 0.001,
        "min_iters": 1,
        "gurobi_settings": {
            "OutputFlag": 0
        }
    }
}

# Load configuration
greedy_experiment = BundleChoice()
greedy_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    # modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
    modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
    errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
    estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    weights = np.abs(np.random.normal(0, 1, (num_items, 10)))
    item_data = {"weights": weights}
    agent_data = {"modular": modular}
    data = {"agent_data": agent_data, 
            "item_data": item_data,
            "errors": errors}
else:
    data = None

# Load and scatter data
greedy_experiment.data.load_and_scatter(data)

# Define features oracle
def features_oracle(i_id, bundle, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][i_id]
    weights = data["item_data"]["weights"]

    if bundle.ndim == 1:
        coverage = (weights * bundle[:,None]).max(0).sum()
        return np.concatenate((modular_agent.T @ bundle, [coverage]))
    else:
        coverage = (weights[:,None,:] * bundle[:,:,None]).max(-1).sum(0)
        return np.vstack((modular_agent.T @ bundle, coverage))

greedy_experiment.features.set_oracle(features_oracle)
theta_0 = np.ones(num_features)
greedy_experiment.subproblems.initialize_local()
obs_bundles = greedy_experiment.subproblems.init_and_solve(theta_0)


# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
    print(obs_bundles.sum(1))
    data["obs_bundle"] = obs_bundles
    data["errors"] = estimation_errors


# greedy_experiment.load_config(cfg)
# greedy_experiment.data.load_and_scatter(data)
# greedy_experiment.features.set_oracle(features_oracle)
# greedy_experiment.subproblems.load()
# tic = datetime.now()
# lambda_k_iter = greedy_experiment.row_generation.solve()
# elapsed = (datetime.now() - tic).total_seconds()
# obj_at_estimate = greedy_experiment.row_generation.objective(lambda_k_iter)
# obj_at_star = greedy_experiment.row_generation.objective(theta_0)
# # Save estimation results as CSV
# if rank == 0:
#     print(f"estimation results:{lambda_k_iter}")
#     print(f"obj at estimate: {obj_at_estimate}")
#     print(f"obj at star: {obj_at_star}")
