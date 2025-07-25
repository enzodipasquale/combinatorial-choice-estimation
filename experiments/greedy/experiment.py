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
num_agents = 500
num_items = 100
num_features = 3
num_simuls = 1

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
        "tol_certificate": 0.001,
        "min_iters": 1,
        "master_settings": {
            "OutputFlag": 1
        }
    }
}

# Load configuration
greedy_experiment = BundleChoice()
greedy_experiment.load_config(cfg)

# Generate data on rank 0
if rank == 0:
    # modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
    modular = np.random.normal(0, 1, (num_agents, num_items, num_features-1))
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items)) 
    agent_data = {"modular": modular}
    data = {"agent_data": agent_data, 
            "errors": errors}
else:
    data = None

# Load and scatter data
greedy_experiment.data.load_and_scatter(data)

# Define features oracle
def features_oracle(i_id, B_j, data):
    """
    Compute features for a given agent and bundle(s).
    Supports both single (1D) and multiple (2D) bundles.
    Returns array of shape (num_features,) for a single bundle,
    or (num_features, m) for m bundles.
    """
    modular_agent = data["agent_data"]["modular"][i_id]
    # modular_item = data["item_data"]["modular"]

    modular_agent = np.atleast_2d(modular_agent)
    # modular_item = np.atleast_2d(modular_item)

    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True

    agent_sum = modular_agent.T @ B_j
    # item_sum = modular_item.T @ B_j
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2

    features = np.vstack((agent_sum, 
                            # item_sum, 
                            neg_sq))
    if single_bundle:
        return features[:, 0] 
    return features

# def features_oracle(i_id, B_j, data):
#     modular_agent = data["agent_data"]["modular"][i_id]
#     if B_j.ndim > 1:
#         raise ValueError("B_j must be a 1D array")
#     agent_sum =  (modular_agent * B_j[:,None]).sum(0)
#     neg_sq = -(B_j.sum() ** 2)

#     features = np.concatenate((agent_sum, [neg_sq]))
    
#     return features

greedy_experiment.features.load(features_oracle)


# Simulate beta_star and generate obs_bundles
beta_star = np.ones(num_features)
# beta_star[-1] = 0
obs_bundles = greedy_experiment.subproblems.init_and_solve(beta_star)

# Estimate parameters using row generation
if rank == 0:
    print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
    print(f"aggregate: {obs_bundles.sum()}")
    data["obs_bundle"] = obs_bundles
    data["errors"] = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    pd.DataFrame(obs_bundles.astype(int)).to_csv(os.path.join(SAVE_PATH, "obs_bundles.csv"), index=False, header=False)
    pd.DataFrame(agent_data["modular"].reshape(-1, num_features-1)).to_csv(os.path.join(SAVE_PATH, "modular.csv"), index=False, header=False)
else:
    data = None

greedy_experiment.load_config(cfg)
greedy_experiment.data.load_and_scatter(data)
greedy_experiment.features.load(features_oracle)
greedy_experiment.subproblems.load()
tic = datetime.now()
lambda_k_iter, p_j_iter = greedy_experiment.row_generation.solve()
elapsed = (datetime.now() - tic).total_seconds()
obj_at_estimate = greedy_experiment.row_generation.ObjVal(lambda_k_iter)
obj_at_star = greedy_experiment.row_generation.ObjVal(beta_star)
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
        "subproblem": greedy_experiment.subproblem_cfg.name,
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
    addOne = 0
    dropOne = 0
    DataArray = np.zeros((num_agents, num_items, num_features))
    for i in range(num_agents):
        features_i_obs = features_oracle(i, obs_bundles[i], data)
        for j in range(num_items):
            alt_bundle = obs_bundles[i].copy()
            if obs_bundles[i,j]:
                alt_bundle[j] = 0
                feat_alt_bundle = features_oracle(i, alt_bundle, data)
                dropOne +=1

            else:
                alt_bundle[j] = 1
                feat_alt_bundle = features_oracle(i, alt_bundle, data)
                addOne +=1 
            Delta_features = features_i_obs - feat_alt_bundle  
            DataArray[i,j,:] = Delta_features

    satisfied_at_star = ((DataArray @ beta_star) >= 0 ).sum()
    print(f"satisfied_at_star: {satisfied_at_star} out of {num_agents * num_items}")


