import os

import numpy as np
import torch 
import yaml
from mpi4py import MPI

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load configuration
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config_simul.yaml")
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data
if rank == 0:  
    num_agents, num_items, num_features = config["num_agents"], config["num_items"], config["num_features"]
    np.random.seed(0)
    num_mod = num_features - 1
    modular_i_j_k = - 10 *  np.random.normal(0, 1, (num_agents, num_items, num_mod)) ** 2
    agent_data = {"modular": modular_i_j_k}
    # quadratic_j_j_k = np.random.choice([0,1], size= (num_items, num_items, num_features - num_mod), p=[0.8, 0.2])
    quadratic_j_j_k = np.exp( - np.random.normal(0, 2, size=(num_items, num_items, num_features - num_mod)) ** 2)
    quadratic_j_j_k *= np.random.choice([0,1], size= (num_items, num_items, num_features - num_mod), p=[0.5, 0.5])
    item_data = {"quadratic":  quadratic_j_j_k}
    num_simuls = config["num_simuls"]
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
    errors *= 10


    data = {
            "agent_data": agent_data,
            "item_data": item_data,
            "errors": errors
            }
else:
    data = None

# User-defined feature oracle
def get_x_k(self, i_id, B_j, local= False):
    modular = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    quadratic = self.item_data["quadratic"]
    return np.concatenate((
                            np.einsum('jk,j->k', modular, B_j),
                            np.einsum('jlk,j,l->k', quadratic, B_j, B_j)
                            ))
        
# Pricing subproblem from bundlechoice library
init_pricing, solve_pricing = get_subproblem(config["subproblem"])


quadsupermod_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
quadsupermod_demo.scatter_data()
quadsupermod_demo.local_data_to_torch()


lambda_k_star = np.ones(config["num_features"]) 
# lambda_k_star = np.array([3.41268895 ,1.33776912, 0.47635377, 2.83031925, 0.98361723])
results = quadsupermod_demo.solve_pricing_offline(lambda_k_star)

# Save results 
if rank == 0:
    obs_bundles = results.astype(bool)
    # print("obs_bundles shape", obs_bundles.shape)
    input_data_path = os.path.join(BASE_DIR, "input_data")
    if not os.path.exists(input_data_path):
        os.makedirs(input_data_path)
    np.save(os.path.join(input_data_path, "obs_bundles.npy"), obs_bundles)
    np.save(os.path.join(input_data_path, "modular.npy"), agent_data["modular"])
    np.save(os.path.join(input_data_path, "quadratic.npy"), item_data["quadratic"])
    # print("Results saved to", input_data_path)
    print("aggregate demands:", obs_bundles.sum(1))


    