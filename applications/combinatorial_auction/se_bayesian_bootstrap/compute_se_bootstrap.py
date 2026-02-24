#!/bin/env python
import sys, os, csv, yaml, json
import numpy as np
from pathlib import Path
from mpi4py import MPI
from datetime import datetime

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bundlechoice import BundleChoice
from bundlechoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.data.prepare_data import main as prepare_data_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR /  "config.yaml"))

app = config.get("application")
boot = config.get("bootstrap")
NUM_BOOTSTRAP = boot.get("num_samples")
BOOT_SEED = boot.get("seed")
ERROR_SEED = app.get("error_seed")
OUTPUT_DIR = APP_DIR / "estimation_results"


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
    n_features = n_id_mod + n_item_mod + n_id_quad + n_item_quad

    dim_cfg = {"n_obs": n_obs, "n_items": n_items, "n_features": n_features}
    id_mod_indices = list(range(n_id_mod))
    id_quad_offset = n_id_mod + n_item_mod
    id_quad_indices = list(range(id_quad_offset, id_quad_offset + n_id_quad))
    item_quad_indices = list(range(n_features - n_item_quad, n_features))
    config["standard_errors"]["parameters_to_log"] = id_mod_indices + id_quad_indices + item_quad_indices
    config["row_generation"]["parameters_to_log"] = id_mod_indices + id_quad_indices + item_quad_indices
    config["dimensions"].update(dim_cfg)
    mod_b = app.get('mod_bounds', {})
    quad_b = app.get('quad_bounds', {})
    quad_id_b = app.get('quad_id_bounds', {})
    updates = {}
    for k in id_mod_indices[1:]:
        updates[k] = (mod_b.get('lb'), mod_b.get('ub'))
    for k in id_quad_indices:
        updates[k] = (quad_id_b.get('lb'), quad_id_b.get('ub'))
    for k in item_quad_indices[:-3]:
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

bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if rank == 0:
    print(f"agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")

callbacks = config.get("callbacks")
# def boot_callback(iter, boot):
#     if boot.comm_manager.is_root() and config.get("constraints", {}).get("pop_dominates_travel"):
#         theta, _ = boot.row_gen.master_variables
#         boot.row_gen.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
#         boot.row_gen.master_model.update()


pt_timeout_cb, _ = adaptive_gurobi_timeout(callbacks['row_gen'])
_, dist_timeout_cb = adaptive_gurobi_timeout(callbacks['boot'])


def boot_callback(iter, boot, master):
    dist_timeout_cb(iter, boot, master)
    if master is not None and iter == 0:
        master.strip_slack_constraints(percentile=callbacks['boot_strip']["percentile"], hard_threshold = callbacks['boot_strip']["hard_threshold"])

checkpoint_dir = str(BASE_DIR)
se_result = bc.standard_errors.compute_distributed_bootstrap(
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "winners_only": app.get("winners_only"),
        "modular_regressors": json.dumps(app.get("modular_regressors", [])),
        "quadratic_regressors": json.dumps(app.get("quadratic_regressors", [])),
        "error_seed": ERROR_SEED,
        "n_obs": bc.n_obs,
        "n_items": bc.n_items,
        "n_features": bc.n_features,
        "num_bootstrap": NUM_BOOTSTRAP,
        "bootstrap_seed": BOOT_SEED,
        "n_samples": se_result.n_samples,
        "confidence": se_result.confidence,
        "pop_dominates_travel": config.get("constraints", {}).get("pop_dominates_travel"),
        "mean_se": float(np.mean(se_result.se)),
        "max_se": float(np.max(se_result.se)),
        "theta_mean": json.dumps(se_result.mean.tolist()),
        "se": json.dumps(se_result.se.tolist()),
        "ci_lower": json.dumps(se_result.ci_lower.tolist()),
        "ci_upper": json.dumps(se_result.ci_upper.tolist()),
        "t_stats": json.dumps(se_result.t_stats.tolist()),
    }
    csv_path = OUTPUT_DIR / "se_bootstrap_runs.csv"
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
