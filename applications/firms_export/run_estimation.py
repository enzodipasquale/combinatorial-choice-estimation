#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

import numpy as np
import yaml
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

### Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Select pricing problem from config
# init_pricing, solve_pricing = get_subproblem(config["subproblem"])
init_pricing, solve_pricing = None, None


### Load data on rank 0
if rank == 0:  
    path = './input_data/'

    obs_bundle = np.load(path + "matching_i_j.npy")

    item_data = {
        "quadratic": np.load(path + "quadratic_characteristic_j_j_k.npy"),
        "weights": np.load(path + "weight_j.npy")
    }

    agent_data = {
        "modular": np.load(path + "modular_characteristics_i_j_k.npy"),
        "capacity": np.load(path + "capacity_i.npy"),
    }

    np.random.seed(0)
    errors = np.random.normal(0, 1, size=(config["num_simuls"], 255, 493))

    data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
else:
    data = None

dims = (255, 493, 4)

### User-defined feature oracle
def compute_features(self, bundle_i_j):
    modular = self.agent_data["modular"]
    quadratic = self.item_data["quadratic"]
    return np.concatenate((
                            np.einsum('ijk,ij->ik', modular, bundle_i_j),
                            np.einsum('jlk,ij,il->ik', quadratic, bundle_i_j, bundle_i_j)
                            ), axis=1)


### Run the estimation
my_test = BundleChoice(data, dims, config, compute_features, init_pricing, solve_pricing)
my_test.scatter_data()
my_test.local_data_to_torch()


# my_test.compute_estimator_row_gen()
print("Rank", rank, "Hi", print(my_test.torch_local_agent_data['modular'].shape))

if rank == 0:
    print(data["agent_data"]["modular"].shape)