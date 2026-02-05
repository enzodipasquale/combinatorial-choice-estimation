#!/bin/env python
import sys, os, csv, json, yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from mpi4py import MPI

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bundlechoice.estimation.callbacks import adaptive_gurobi_timeout
from bundlechoice import BundleChoice
from applications.combinatorial_auction.data.prepare_data import main as prepare_data_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config.get("application", {})
DELTA, WINNERS_ONLY, HQ_DISTANCE = app.get("delta"), app.get("winners_only"), app.get("hq_distance")
CONTINENTAL_ONLY = app.get("continental_only", False)
INCLUDE_ADJACENCY = app.get("adjacency", False)
ERROR_SEED = app.get("error_seed")
OUTPUT_DIR = APP_DIR / "estimation_results"


input_data = prepare_data_main(
    delta=DELTA,
    winners_only=WINNERS_ONLY,
    hq_distance=HQ_DISTANCE,
)

n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
n_quad_item = input_data["item_data"]["quadratic"].shape[-1]
n_mod_agent = input_data["id_data"]["modular"].shape[-1]
n_mod_item = input_data["item_data"]["modular"].shape[-1]
n_features = n_quad_item + n_mod_agent + n_mod_item
config["dimensions"]["n_obs"] = n_obs
config["dimensions"]["n_items"] = n_items   
config["dimensions"]["n_features"] = n_features   

print(n_obs, n_features )

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation"]})
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

callbacks = config.get("callbacks", {})
adaptive_cfg = callbacks.get("adaptive_timeout", {})
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

if rank == 0 and app.get("save_results", True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "delta": DELTA,
        "winners_only": WINNERS_ONLY,
        "hq_distance": HQ_DISTANCE,
        "error_seed": ERROR_SEED,
        "n_obs": bc.n_obs,
        "n_items": bc.n_items,
        "n_features": bc.n_features,
        "n_simulations": bc.n_simulations,
        "num_mpi": comm.Get_size(),
        "converged": result.converged,
        "num_iterations": result.num_iterations,
        "total_time": result.total_time,
        "final_objective": result.final_objective,
        "n_constraints": result.n_constraints,
        "final_reduced_cost": result.final_reduced_cost,
        "final_n_violations": result.final_n_violations,
        "theta_hat": json.dumps(result.theta_hat.tolist()),
    }
    csv_path = OUTPUT_DIR / "point_estimate_runs.csv"
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
