import os

import numpy as np
import numpy as np
import yaml
from mpi4py import MPI

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Select pricing problem from config
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config_simul.yaml")
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

### Load data on rank 0
if rank == 0:  
    num_agents, num_items, num_features = config["num_agents"], config["num_items"], config["num_features"]
    np.random.seed(0)
    agent_data = {
                 "modular": np.random.normal(0, 1, (num_agents, num_items, num_features-1))**2
                }
    num_simuls = config["num_simuls"]
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

    data = {
            "agent_data": agent_data,
            "errors": errors
            }
else:
    data = None

def get_x_k(self, i_id, B_j, local= False):  
    modular = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    return np.concatenate((modular[B_j].sum(0), [-B_j.sum() **(1.5)]))

greedy_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
greedy_demo.scatter_data()
lambda_k_star = np.ones(config["num_features"]) 
results = greedy_demo.solve_pricing_offline(lambda_k_star)

# Save results 
if rank == 0:
    obs_bundles = results[:, -num_items:].astype(bool)
    input_data_path = os.path.join(BASE_DIR, "input_data")
    if not os.path.exists(input_data_path):
        os.makedirs(input_data_path)
    np.save(os.path.join(input_data_path, "obs_bundles.npy"), obs_bundles)
    np.save(os.path.join(input_data_path, "modular.npy"), agent_data["modular"])
    print("Results saved to", input_data_path)
    print(obs_bundles.sum(1))


