#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

import numpy as np
import yaml
from mpi4py import MPI
import os
import platform

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "input_data")

IS_LOCAL = os.path.exists("/Users/enzo-macmini") or os.path.exists("/Users/enzo-macbookpro") 
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Select pricing problem from config
init_pricing, solve_pricing = get_subproblem(config["subproblem"])

### Load data on rank 0
if rank == 0:  
    path = './input_data/'

    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    item_data = {
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }

    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
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
my_test.compute_estimator_row_gen()
