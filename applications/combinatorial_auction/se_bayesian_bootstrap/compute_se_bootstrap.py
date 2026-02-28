#!/bin/env python
import sys, os, yaml
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
from applications.combinatorial_auction.results import save_bootstrap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR /  "config.yaml"))

def _get_int_env(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else int(default)


app = config.get("application", {})
boot = config.get("bootstrap", {})
dims = config.get("dimensions", {})

NUM_BOOTSTRAP = _get_int_env("NUM_SAMPLES", boot.get("num_samples"))
N_SIMULATIONS = _get_int_env("N_SIMULATIONS", dims.get("n_simulations"))

config.setdefault("dimensions", {})["n_simulations"] = N_SIMULATIONS

BOOT_SEED = boot.get("seed")
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
    updates = {}
    for k in range(1, n_id_mod):
        updates[k] = (mod_b.get('lb'), mod_b.get('ub'))
    for k in range(id_quad_offset, id_quad_offset + n_id_quad):
        updates[k] = (quad_id_b.get('lb'), quad_id_b.get('ub'))
    for k in range(item_quad_offset, n_covariates - 3):
        updates[k] = (quad_b.get('lb'), quad_b.get('ub'))
    for bounds in [config["row_generation"]["theta_bounds"], config["standard_errors"]["theta_bounds"]]:
        for k, (lb, ub) in updates.items():
            if lb is not None:
                bounds["lbs"][k] = lb
            if ub is not None:
                bounds["ubs"][k] = ub

else:
    input_data = None

config = comm.bcast(config, root=0)

auction = cc.Model()
auction.load_config(config)
auction.data.load_and_distribute_input_data(input_data)
auction.features.build_quadratic_covariates_from_data()
auction.features.build_local_modular_error_oracle(seed=ERROR_SEED)
auction.subproblems.load_solver()

if rank == 0:
    print(f"agents={auction.n_obs}, items={auction.n_items}, bootstrap={NUM_BOOTSTRAP}")

callbacks = config.get("callbacks")
pt_timeout_cb, _ = adaptive_gurobi_timeout(callbacks['row_gen'])
_, dist_timeout_cb = adaptive_gurobi_timeout(callbacks['boot'])


def boot_callback(iter, boot, master):
    dist_timeout_cb(iter, boot, master)
    if master is not None and iter == 0:
        master.strip_slack_constraints(percentile=callbacks['boot_strip']["percentile"], hard_threshold = callbacks['boot_strip']["hard_threshold"])

checkpoint_dir = str(BASE_DIR)
se_result = auction.standard_errors.compute_distributed_bootstrap(
    num_bootstrap=NUM_BOOTSTRAP,
    seed=BOOT_SEED,
    verbose=True,
    pt_estimate_callbacks=(None, pt_timeout_cb),
    bootstrap_callback=boot_callback,
    method='bayesian',
    save_model_dir=checkpoint_dir,
    load_model_dir=checkpoint_dir,
)



if rank == 0 and se_result is not None and app.get("save_results", True):
    save_bootstrap(config, se_result, auction.n_obs, auction.n_items, auction.n_covariates,
                   NUM_BOOTSTRAP, BOOT_SEED)
