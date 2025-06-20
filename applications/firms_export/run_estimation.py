#!/bin/env python

from bundlechoice import BundleChoice
from bundlechoice.subproblems import get_subproblem

import numpy as np
import yaml
from mpi4py import MPI
import os


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
                    "modular": np.load(os.path.join(INPUT_DIR, "modular_j_k.npy")),
                    "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_j_j_k.npy"))
                }

    agent_data = {
                    "modular": np.load(os.path.join(INPUT_DIR, "modular_i_j_k.npy"))
                }

    np.random.seed(34254)
    shape = (config["num_simuls"], config["num_agents"], config["num_items"])
    errors = np.random.normal(0, 1, size=shape)

    # Correlation structure
    # rho_0 = .16
    # rho_d = .82
    # sigmasq = 18
    # distance_j_j = np.load(os.path.join(INPUT_DIR, "distance_j_j.npy"))
    # Covariance = sigmasq * rho_0 * np.exp(- rho_d * distance_j_j)
    # from scipy.linalg import sqrtm
    # cov_sqrt = sqrtm(Covariance)
    # for s in range(config["num_simuls"]):
    #     for i in range(config["num_agents"]):
    #         errors[s, i] = cov_sqrt @ errors[s, i]

    # Blocking shocks
    # p = 0.66
    # random_vals = np.random.rand(*shape)
    # max_val = ((item_data["quadratic"] @ np.ones(2) * 200).sum() + 200* agent_data["modular"].sum((1,2))).max()
    # print("max_val:", max_val)
    # errors = np.where(random_vals < p, errors, - float('inf'))

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
    modular_agent = self.local_agent_data["modular"][i_id] if local else self.agent_data["modular"][i_id]
    modular_item = self.item_data["modular"]
    quadratic_item = self.item_data["quadratic"]
    return np.concatenate(( np.einsum('jk,j->k', modular_item, B_j),
                            np.einsum('jk,j->k', modular_agent, B_j),
                            np.einsum('jlk,j,l->k', quadratic_item, B_j, B_j)
                            ))
        
# Demand orable from library
init_pricing, solve_pricing = get_subproblem(config["subproblem_name"])

# Run estimation
firms_export = BundleChoice(data, config, get_x_k, init_pricing, solve_pricing)
firms_export.scatter_data()
firms_export.local_data_to_torch()
firms_export.compute_estimator_row_gen()



#HPC
# ========== Final Solution ==========
# lambda_1 = 1.059610474320977
# lambda_2 = 0.10969382797095822
# lambda_3 = 0.10037003252260504
# lambda_4 = 1.5651111150566515
# lambda_5 = -0.8273214069366847
# lambda_6 = 0.5552463235956646