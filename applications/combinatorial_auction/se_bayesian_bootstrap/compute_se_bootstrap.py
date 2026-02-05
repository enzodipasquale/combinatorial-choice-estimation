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

from bundlechoice import BundleChoice
from bundlechoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.data.prepare_data import main as prepare_data_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config = yaml.safe_load(open(BASE_DIR /  "config.yaml"))

app = config.get("application", {})
boot = config.get("bootstrap", {})
DELTA = app.get("delta", 4)
WINNERS_ONLY = app.get("winners_only", False)
HQ_DISTANCE = app.get("hq_distance", False)
CONTINENTAL_ONLY = app.get("continental_only", False)
INCLUDE_ADJACENCY = app.get("adjacency", False)
NUM_BOOTSTRAP, SEED = boot.get("num_samples", 100), boot.get("seed", 1995)
ERROR_SEED = app.get("error_seed", 1995)
OUTPUT_DIR = APP_DIR / "estimation_results"


if rank == 0:
    input_data = prepare_data_main(
        delta=DELTA,
        winners_only=WINNERS_ONLY,
        hq_distance=HQ_DISTANCE,
        continental_only=CONTINENTAL_ONLY,
        include_adjacency=INCLUDE_ADJACENCY,
    )
else:
    input_data = None


bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]})
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
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")


def strip_master_constraints(boot, rowgen):
    strip_cfg = callbacks.get("strip")
    percentile = strip_cfg.get("percentile")
    hard_threshold = strip_cfg.get("hard_threshold")
    rowgen.strip_slack_constraints(percentile=percentile, hard_threshold=hard_threshold)


adaptive_cfg = callbacks.get("adaptive_timeout")
timeout_callback = adaptive_gurobi_timeout(
    initial_timeout=adaptive_cfg.get("initial"),
    final_timeout=adaptive_cfg.get("final"),
    transition_iterations=adaptive_cfg.get("transition_iterations"),
    strategy=adaptive_cfg.get("strategy", "step")
)
se_result = bc.standard_errors.compute_bootstrap(num_bootstrap=NUM_BOOTSTRAP, 
                                                                seed=SEED, 
                                                                verbose=True,
                                                                bootstrap_callback=strip_master_constraints,
                                                                row_gen_iteration_callback=timeout_callback,
                                                                method= 'bayesian')



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
        "bootstrap_seed": SEED,
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