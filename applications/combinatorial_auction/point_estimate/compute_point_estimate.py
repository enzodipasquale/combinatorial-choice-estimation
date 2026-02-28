#!/bin/env python
import sys, yaml
import numpy as np
from pathlib import Path
from mpi4py import MPI

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combchoice as cc
from combchoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.data.prepare_data import main as prepare_data_main
from applications.combinatorial_auction.results import save_point_estimate

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))

app = config.get("application")
ERROR_SEED = app.get("error_seed")

if rank == 0:
    input_data = prepare_data_main(
        winners_only=app.get("winners_only", False),
        continental_only=app.get("continental_only"),
        rescale_features=app.get("rescale_features"),
        modular_regressors=app.get("modular_regressors"),
        quadratic_regressors=app.get("quadratic_regressors"),
        quadratic_id_regressors=app.get("quadratic_id_regressors"),
    )
    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = input_data["item_data"]["modular"].shape[-1]
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_covariates = n_id_mod + n_item_mod + n_id_quad + n_item_quad

    dim_cfg = {"n_obs": n_obs, "n_items": n_items, "n_covariates": n_covariates}
    id_mod_indices = list(range(n_id_mod))
    id_quad_offset = n_id_mod + n_item_mod
    id_quad_indices = list(range(id_quad_offset, id_quad_offset + n_id_quad))
    item_quad_indices = list(range(n_covariates - n_item_quad, n_covariates))
    config["row_generation"]["parameters_to_log"] = id_mod_indices + id_quad_indices + item_quad_indices
    config["dimensions"].update(dim_cfg)
    mod_b = app.get('mod_bounds', {})
    quad_b = app.get('quad_bounds', {})
    quad_id_b = app.get('quad_id_bounds', {})
    bounds = config["row_generation"]["theta_bounds"]
    for k in id_mod_indices[1:]:
        if mod_b.get('lb') is not None:
            bounds["lbs"][k] = mod_b['lb']
        if mod_b.get('ub') is not None:
            bounds["ubs"][k] = mod_b['ub']
    for k in id_quad_indices:
        if quad_id_b.get('lb') is not None:
            bounds["lbs"][k] = quad_id_b['lb']
        if quad_id_b.get('ub') is not None:
            bounds["ubs"][k] = quad_id_b['ub']
    for k in item_quad_indices[:-3]:
        if quad_b.get('lb') is not None:
            bounds["lbs"][k] = quad_b['lb']
        if quad_b.get('ub') is not None:
            bounds["ubs"][k] = quad_b['ub']
else:
    input_data = None

config = comm.bcast(config, root=0)

auction = cc.Model()
auction.load_config(config)
auction.data.load_and_distribute_input_data(input_data)
auction.features.build_quadratic_covariates_from_data()
auction.features.build_local_modular_error_oracle(seed=ERROR_SEED)
auction.subproblems.load_solver()

if config.get("constraints", {}).get("pop_dominates_travel"):
    def custom_constraint(row_gen_manager):
        theta, u = row_gen_manager.master_variables
        row_gen_manager.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
        row_gen_manager.master_model.update()
    auction.config.row_generation.initialization_callback = custom_constraint

callbacks = config.get("callbacks")

if rank == 0:
    print(f"agents={auction.n_obs}, items={auction.n_items}, features={auction.n_covariates}")

pt_timeout_cb, _ = adaptive_gurobi_timeout(callbacks['row_gen'])

# Pass it to solve
result = auction.row_generation.solve(
    iteration_callback=pt_timeout_cb,
    verbose=True
)

if rank == 0:
    print(result.theta_hat)
    print(result.theta_hat[1:-3].max())

if rank == 0 and result is not None and app.get("save_results", True):
    save_point_estimate(config, result, auction.n_obs, auction.n_items, auction.n_covariates)