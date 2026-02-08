#!/bin/env python
import sys, os, csv, yaml, json
import numpy as np
from pathlib import Path
from mpi4py import MPI
import datetime

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bundlechoice import BundleChoice
from bundlechoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.data.prepare_data import main as prepare_data_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))

app = config.get("application")
DELTA = app.get("delta")
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

    dim_cfg = {"n_obs": n_obs, "n_items": n_items, "n_features": n_features}
else:
    input_data = None
    dim_cfg = None

dim_cfg = comm.bcast(dim_cfg, root=0)
config["dimensions"].update(dim_cfg)

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation"]})
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if config.get("constraints", {}).get("pop_dominates_travel"):
    def custom_constraint(row_gen_manager):
        theta, u = row_gen_manager.master_variables
        row_gen_manager.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
        row_gen_manager.master_model.update()
    bc.config.row_generation.initialization_callback = custom_constraint

callbacks = config.get("callbacks")

if rank == 0:
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, features={bc.n_features}")

adaptive_cfg = callbacks.get("adaptive_timeout")
timeout_callback = adaptive_gurobi_timeout(
    initial_timeout=adaptive_cfg.get("initial"),
    final_timeout=adaptive_cfg.get("final"),
    transition_iterations=adaptive_cfg.get("transition_iterations"),
    strategy=adaptive_cfg.get("strategy", "step")
)

# Pass it to solve
result = bc.row_generation.solve(
    iteration_callback=timeout_callback,
    verbose=True
)

if rank == 0:
    print(result.theta_hat)
    print(result.theta_hat[1:-3].max())

if rank == 0 and result is not None and app.get("save_results", True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "delta": DELTA,
        "winners_only": app.get("winners_only"),
        "hq_distance": app.get("hq_distance"),
        "error_seed": ERROR_SEED,
        "n_obs": bc.n_obs,
        "n_items": bc.n_items,
        "n_features": bc.n_features,
        "pop_dominates_travel": config.get("constraints", {}).get("pop_dominates_travel"),
        "theta": json.dumps(result.theta.tolist()),
    }
    csv_path = OUTPUT_DIR / "point_estimate_runs.csv"
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)