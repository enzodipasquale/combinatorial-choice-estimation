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
DELTA = app.get("delta")
NUM_BOOTSTRAP = boot.get("num_samples")
BOOT_SEED = boot.get("seed")
ERROR_SEED = app.get("error_seed")
OUTPUT_DIR = APP_DIR / "estimation_results"


if rank == 0:
    input_data = prepare_data_main(
        delta=DELTA,
        winners_only=app.get("winners_only", False),
        hq_distance=app.get("hq_distance", False),
        continental_only=app.get("continental_only")
    )
    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = input_data["item_data"]["modular"].shape[-1]
    n_features = n_item_quad + n_id_mod + n_item_mod

    dim_cfg = {"n_obs":n_obs, "n_items":n_items, "n_features":n_features}
else:
    input_data = None
    dim_cfg = None

dim_cfg = comm.bcast(dim_cfg, root = 0)
config["dimensions"].update(dim_cfg)

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]})
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if rank == 0:
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")

callbacks = config.get("callbacks")
def boot_callback(iter, boot):
    if boot.comm_manager.is_root() and config.get("constraints", {}).get("pop_dominates_travel"):
        theta, _ = boot.row_gen.master_variables
        boot.row_gen.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
        boot.row_gen.master_model.update()
    if iter > 0:
        strip_cfg = callbacks.get("strip")
        percentile = strip_cfg.get("percentile")
        hard_threshold = strip_cfg.get("hard_threshold")
        boot.row_gen.strip_slack_constraints(percentile=percentile, hard_threshold=hard_threshold)


adaptive_cfg = callbacks.get("adaptive_timeout")
timeout_callback = adaptive_gurobi_timeout(
    initial_timeout=adaptive_cfg.get("initial"),
    final_timeout=adaptive_cfg.get("final"),
    transition_iterations=adaptive_cfg.get("transition_iterations"),
    strategy=adaptive_cfg.get("strategy", "step")
)
# se_result = bc.standard_errors.compute_bootstrap(num_bootstrap=NUM_BOOTSTRAP, 
#                                                                 seed=BOOT_SEED, 
#                                                                 verbose=True,
#                                                                 bootstrap_callback=boot_callback,
#                                                                 row_gen_iteration_callback=timeout_callback,
#                                                                 method= 'bayesian')

checkpoint_dir = str(BASE_DIR)
se_result = bc.standard_errors.compute_distributed_bootstrap(
    num_bootstrap=NUM_BOOTSTRAP,
    seed=BOOT_SEED,
    verbose=True,
    row_gen_iteration_callback=timeout_callback,
    method='bayesian',
    save_model_dir=checkpoint_dir,
    load_model_dir=checkpoint_dir,
)



if rank == 0 and se_result is not None and app.get("save_results", True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "delta": DELTA,
        "winners_only": app.get("winners_only"),
        "hq_distance": app.get("hq_distance"),
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
