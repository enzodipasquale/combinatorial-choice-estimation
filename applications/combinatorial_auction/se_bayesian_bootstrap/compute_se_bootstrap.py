#!/bin/env python
import sys, os, csv, yaml
import numpy as np
from pathlib import Path
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
NUM_BOOTSTRAP, SEED = boot.get("num_samples", 100), boot.get("seed", 1995)
ERROR_SEED = app.get("error_seed", 1995)
OUTPUT_DIR = APP_DIR / "estimation_results"


bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]})

if rank == 0:
    input_data = prepare_data_main(
        delta=DELTA,
        winners_only=app.get("winners_only", False),
        hq_distance=app.get("hq_distance", False)
    )
else:
    input_data = None

bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if theta_bounds := config.get("theta_bounds"):
    theta_lbs = np.zeros(bc.n_features)
    theta_ubs = np.ones(bc.n_features) * 2000
    for k,v in theta_bounds['lbs'].items():
        theta_lbs[k] = v
    for k,v in theta_bounds['ubs'].items():
        theta_ubs[k] = v     
    bc.config.row_generation.theta_lbs = theta_lbs
    bc.config.row_generation.theta_ubs = theta_ubs

if config.get("constraints", {}).get("pop_dominates_travel"):
    def custom_constraint(row_gen_manager):
        theta, u = row_gen_manager.master_variables
        row_gen_manager.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
        row_gen_manager.master_model.update()
    bc.config.row_generation.initialization_callback = custom_constraint

callbacks = config.get("callbacks", {})
if adaptive_cfg := callbacks.get("adaptive_timeout"):
    bc.config.row_generation.subproblem_callback = adaptive_gurobi_timeout(
        initial_timeout=adaptive_cfg.get("initial", 1.0),
        final_timeout=adaptive_cfg.get("final", 30.0),
        transition_iterations=adaptive_cfg.get("transition_iterations", 15),
        strategy = adaptive_cfg.get("strategy", "step")
    )


if rank == 0:
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")


def strip_master_constraints(boot, rowgen):
    strip_cfg = callbacks.get("strip", {})
    percentile = strip_cfg.get("percentile", 50)
    hard_threshold = strip_cfg.get("hard_threshold", 2)
    rowgen.strip_slack_constraints(percentile=percentile, hard_threshold=hard_threshold)

adaptive_cfg = callbacks.get("adaptive_timeout", {})
timeout_callback = adaptive_gurobi_timeout(
    initial_timeout=adaptive_cfg.get("initial", 1.0),
    final_timeout=adaptive_cfg.get("final", 1.0),
    transition_iterations=adaptive_cfg.get("transition_iterations", 10),
    strategy=adaptive_cfg.get("strategy", "linear")
)
se_result = bc.standard_errors.compute_bayesian_bootstrap(num_bootstrap=NUM_BOOTSTRAP, seed=SEED, verbose=True,
                                                     bootstrap_callback=lambda self, rowgen: strip_master_constraints(boot, rowgen),
                                                     row_gen_iteration_callback=timeout_callback)



# if rank == 0 and se_result is not None:
#     theta_point = theta_hat
    
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     np.savez(OUTPUT_DIR / "se_bootstrap.npz", mean=se_result.mean, se=se_result.se, 
#              ci_lower=se_result.ci_lower, ci_upper=se_result.ci_upper)
    
#     row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "delta": DELTA,
#            "winners_only": app.get("winners_only"), "hq_distance": app.get("hq_distance"),
#            "n_obs": bc.n_obs, "n_items": bc.n_items, "n_features": bc.n_features,
#            "num_bootstrap": NUM_BOOTSTRAP, "seed": SEED,
#            "converged": result.converged, "num_iterations": result.num_iterations}
   
    
#     csv_path = OUTPUT_DIR / "se_bootstrap.csv"
#     write_header = not csv_path.exists()
#     with open(csv_path, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=row.keys())
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)
    
#     print("\nResults:")

comm.Barrier()
