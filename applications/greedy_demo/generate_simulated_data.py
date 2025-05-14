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
    np.random.seed(0)
    agent_data = {
                 "modular": np.random.normal(0, 1, (config["num_agents"], config["num_items"], config["num_features"]-1))
                }
    errors = np.random.normal(0, 1, size=(config["num_simuls"], config["num_agents"], config["num_items"]))

    data = {
            "agent_data": agent_data,
            "errors": errors
            }
else:
    data = None

def get_x_k(self, i_id, B_j):  
    x_i_j_k = self.agent_data["modular"]  
    return np.concatenate((x_i_j_k[i_id, B_j].sum(1), [B_j.sum(0) **2]))

def get_x_i_k(self, bundle_i_j):
    modular = self.agent_data["modular"]
    return np.concatenate((
                            np.einsum('ijk,ij->ik', modular, bundle_i_j),
                            bundle_i_j.sum(1, keepdims=True) ** 2
                            ), axis=1)

greedy_demo = BundleChoice(data, config, get_x_i_k, init_pricing, solve_pricing)
greedy_demo.scatter_data()
lambda_k_star = np.ones(config["num_features"])
greedy_demo.solve_all_pricing(lambda_k_star)
