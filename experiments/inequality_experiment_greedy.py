import numpy as np
from bundlechoice.core import BundleChoice
from datetime import datetime

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


modular = np.abs(np.random.normal(0, 1, (num_agents, num_items, num_features-1)))
errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
agent_data = {"modular": modular}
data = {"agent_data": agent_data, 
        "errors": errors}
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

print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).max()}")
print(f"aggregate: {obs_bundles.sum()}")
data["obs_bundle"] = obs_bundles
data["errors"] = estimation_errors

inequality_experiment.load_config(cfg)
inequality_experiment.data.load_and_scatter(data)
inequality_experiment.features.set_oracle(features_oracle)
inequality_experiment.subproblems.load()

tic = datetime.now()
theta_hat = inequality_experiment.inequalities.solve()
elapsed = (datetime.now() - tic).total_seconds()

# Print estimation results
print(f"estimation results: {theta_hat}")
print(f"true theta: {theta_0}")
print(f"elapsed time: {elapsed:.2f} seconds")
    
