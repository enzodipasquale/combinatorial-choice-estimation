import os
import numpy as np
import yaml
from mpi4py import MPI

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)


# Load data on rank 0
if rank == 0:  
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")

    obs_bundle = np.load(os.path.join(INPUT_DIR, "obs_bundles.npy"))
       item_data = {
                        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
                    }

    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity.npy")),
                }
                
    np.random.seed(42898)
    errors = np.random.normal(0, 1, size=(config["num_simuls"], config["num_agents"], config["num_items"]))

    data = {
            "item_data": item_data,
            "agent_data": agent_data,
            "errors": errors,
            "obs_bundle": obs_bundle
            }
else:
    data = None

# User-defined feature oracle
def get_x_k(self, i_id, B_j, local= False):  
    modular = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    return modular[B_j].sum(0)

# Demand orable from library
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

# Create the BundleChoice instance
knapsack_demo = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
knapsack_demo.scatter_data()
# knapsack_demo.compute_estimator_row_gen()

