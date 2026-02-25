#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from bundlechoice import BundleChoice
from bundlechoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.counterfactuals.MTA_licenses.prepare_data_counterfactual import main as prepare_mta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ESTIMATION_DIR = APP_DIR / "estimation_results"
USE_BOOTSTRAP = False  # True: se_bootstrap_runs.csv, False: point_estimate_runs.csv
source = "se_bootstrap_runs.csv" if USE_BOOTSTRAP else "point_estimate_runs.csv"


def load_estimation_result(source: str, run_idx: int = -1):
    path = ESTIMATION_DIR / source
    df = __import__("pandas").read_csv(path)
    row = df.iloc[run_idx]
    theta = np.array(json.loads(row["theta_mean"] if "theta_mean" in row else row["theta_hat"]))
    return {
        "theta": theta,
        "delta": int(row["delta"]),
        "winners_only": bool(row["winners_only"]),
        "hq_distance": bool(row.get("hq_distance", False)),
        "n_features": int(row["n_features"]),
        "error_seed": int(row.get("error_seed", 1996)),
    }

est = load_estimation_result(source)
include_adjacency = est["n_features"] == 498
n_quad = 4 if include_adjacency else 3

if rank == 0:
    mta_data = prepare_mta(delta=est["delta"], winners_only=est["winners_only"], continental_only=True, include_adjacency=include_adjacency)
    id_data, item_data = mta_data["id_data"], mta_data["item_data"]
    n_obs = id_data["obs_bundles"].shape[0]
    n_mod_agent, n_mod_item = id_data["modular"].shape[-1], item_data["modular"].shape[-1]
else:
    mta_data, n_obs, n_mod_agent, n_mod_item = None, 0, 0, 0

n_obs, n_mod_agent, n_mod_item = comm.bcast((n_obs, n_mod_agent, n_mod_item), root=0)
n_items = n_mod_item - n_quad
n_features = n_mod_agent + n_mod_item + n_quad

theta_est = est["theta"]
theta_init = np.zeros(n_features)
theta_init[:n_mod_agent] = theta_est[:n_mod_agent]
theta_init[n_mod_agent + n_items:] = np.tile(theta_est[-n_quad:], 2)

bc = BundleChoice()
cfg = {"dimensions": {"n_obs": n_obs, "n_items": n_items, "n_features": n_features, "n_simulations": 20},
       "subproblem": {"name": "QuadraticKnapsackGRB", "GRB_Params": {"TimeLimit": 1.0}},
       "row_generation": {"max_iters": 200, "tolerance": 0.01, "parameters_to_log": [4,5,6,7,8],
                         "theta_bounds": {"lb": 0, "ub": 10000}}}
bc.load_config(cfg)

bc.data.load_and_distribute_input_data(mta_data if rank == 0 else None)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=est["error_seed"])
bc.subproblems.load_solver()

def fix_theta(row_gen):
    if rank != 0 or row_gen.master_model is None:
        return
    theta, _ = row_gen.master_variables
    for i in range(theta.size):
        theta[i].Start = theta_init[i]
    fixed = list(range(n_mod_agent)) + list(range(n_mod_agent + n_items, n_features))
    for i in fixed:
        theta[i].LB = theta[i].UB = theta_init[i]
    row_gen.master_model.update()

pt_timeout_cb, _ = adaptive_gurobi_timeout([
    {'iters': 30, 'timeout': 1.0},
    {'timeout': 10.0},
])
bc.row_generation.solve(initialization_callback=fix_theta, iteration_callback=pt_timeout_cb, verbose=True)
