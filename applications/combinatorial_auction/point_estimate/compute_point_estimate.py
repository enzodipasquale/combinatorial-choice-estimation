#!/bin/env python
import json, sys, yaml
import numpy as np
from pathlib import Path
from mpi4py import MPI

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout
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

    modular_regressors = app.get("modular_regressors", [])
    quadratic_id_regressors = app.get("quadratic_id_regressors", [])
    quadratic_regressors = app.get("quadratic_regressors", [])
    covariate_names = {}
    for i, name in enumerate(modular_regressors):
        covariate_names[i] = name
    id_quad_offset = n_id_mod + n_item_mod
    for i, name in enumerate(quadratic_id_regressors):
        covariate_names[id_quad_offset + i] = name
    item_quad_offset = id_quad_offset + n_id_quad
    for i, name in enumerate(quadratic_regressors):
        covariate_names[item_quad_offset + i] = name

    config["dimensions"].update({
        "n_obs": n_obs, "n_items": n_items,
        "n_covariates": n_covariates, "covariate_names": covariate_names,
    })

    mod_b = app.get('mod_bounds', {})
    quad_b = app.get('quad_bounds', {})
    quad_id_b = app.get('quad_id_bounds', {})
    bounds = config["row_generation"]["theta_bounds"]
    for k in range(1, n_id_mod):
        if mod_b.get('lb') is not None:
            bounds["lbs"][k] = mod_b['lb']
        if mod_b.get('ub') is not None:
            bounds["ubs"][k] = mod_b['ub']
    for k in range(id_quad_offset, id_quad_offset + n_id_quad):
        if quad_id_b.get('lb') is not None:
            bounds["lbs"][k] = quad_id_b['lb']
        if quad_id_b.get('ub') is not None:
            bounds["ubs"][k] = quad_id_b['ub']
    for k in range(item_quad_offset, n_covariates - 3):
        if quad_b.get('lb') is not None:
            bounds["lbs"][k] = quad_b['lb']
        if quad_b.get('ub') is not None:
            bounds["ubs"][k] = quad_b['ub']
else:
    input_data = None

config = comm.bcast(config, root=0)

auction = ce.Model()
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

if rank == 0 and result is not None:
    out = {
        "theta_hat": result.theta_hat.tolist(),
        "n_items": n_items, "n_obs": n_obs, "n_covariates": n_covariates,
        "n_id_mod": n_id_mod, "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad, "n_item_quad": n_item_quad,
        "specification": {
            "modular": modular_regressors,
            "quadratic": quadratic_regressors,
            "quadratic_id": quadratic_id_regressors,
        },
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
    }
    with open(BASE_DIR / "bta_estimation_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {BASE_DIR / 'bta_estimation_result.json'}")