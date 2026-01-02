import numpy as np
from bundlechoice.core import BundleChoice
from datetime import datetime

# Define dimensions
num_agents = 1000
num_items = 30
num_simulations = 1
modular_agent_features = 2
quadratic_item_features = 0
num_features = modular_agent_features + quadratic_item_features
sigma = 1

# Define configuration as a dictionary
cfg = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simulations": num_simulations,
    },
    "subproblem": {
        "name": "QuadSupermodularNetwork",
    }
}

# Load configuration
inequality_experiment = BundleChoice()
inequality_experiment.load_config(cfg)

# Generate data
# Modular agent features
modular_agent = -5 * np.random.normal(0, 1, (num_agents, num_items, modular_agent_features)) ** 2 
agent_data = {"modular": modular_agent}

# Quadratic item features
quadratic_item = 0.5 * np.exp(-np.random.normal(0, 2, size=(num_items, num_items, quadratic_item_features)) ** 2)
quadratic_item *= (1 - np.eye(num_items, dtype=int))[:,:, None]
item_data = {"quadratic": quadratic_item}

# Errors
errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items)) 
estimation_errors = sigma * np.random.normal(0, 1, size=(num_simulations, num_agents, num_items))

# Data
data = {"agent_data": agent_data, 
        # "item_data": item_data, 
        "errors": errors}

# Load and scatter data
inequality_experiment.data.load_and_scatter(data)
inequality_experiment.features.build_from_data()

theta_0 = np.ones(num_features)
obs_bundles, _ = inequality_experiment.subproblems.init_and_solve(theta_0, return_values=True)

# Estimate parameters using inequalities method
print(f"aggregate demands: {obs_bundles.sum(1).min()}, {obs_bundles.sum(1).mean()}, {obs_bundles.sum(1).max()}")
print(f"demands: {obs_bundles.sum(1)}")
print(f"aggregate: {obs_bundles.sum()}")
data["obs_bundle"] = obs_bundles
data["errors"] = estimation_errors

inequality_experiment.load_config(cfg)
inequality_experiment.data.load_and_scatter(data)
inequality_experiment.features.build_from_data()
inequality_experiment.subproblems.load()

tic = datetime.now()
result = inequality_experiment.inequalities.solve()
elapsed = (datetime.now() - tic).total_seconds()
theta_hat = result.theta_hat

# Print estimation results
print(f"estimation results: {theta_hat}")
print(f"true theta: {theta_0}")
print(f"elapsed time: {elapsed:.2f} seconds")


# inequality_experiment.row_generation.solve()