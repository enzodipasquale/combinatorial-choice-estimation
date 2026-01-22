#!/bin/env python
import sys, os, csv, yaml
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

IS_LOCAL = Path("/Users/enzo-macbookpro").exists()
config = yaml.safe_load(open(BASE_DIR / ("config_local.yaml" if IS_LOCAL else "config.yaml")))

app = config.get("application", {})
boot = config.get("bootstrap", {})
DELTA = app.get("delta", 4)
NUM_BOOTSTRAP, SEED = boot.get("num_samples", 200), boot.get("seed", 1995)
ERROR_SEED = app.get("error_seed", 1995)
OUTPUT_DIR = APP_DIR / "estimation_results"




bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]})

if rank == 0:
    input_data = prepare_data_main(
        delta=DELTA,
        winners_only=app.get("winners_only", False),
        hq_distance=app.get("hq_distance", False),
        save_data=False
    )
else:
    input_data = None

bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if theta_bounds := config.get("theta_bounds"):
    theta_lbs = np.zeros(bc.n_features)
    if "air_travel_lb" in theta_bounds:
        theta_lbs[-1] = theta_bounds["air_travel_lb"]
    if "travel_survey_lb" in theta_bounds:
        theta_lbs[-2] = theta_bounds["travel_survey_lb"]
    bc.config.row_generation.theta_lbs = theta_lbs

if config.get("constraints", {}).get("pop_dominates_travel"):
    def add_constraint(row_gen_manager):
        theta, u = row_gen_manager.master_variables
        row_gen_manager.master_model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
        row_gen_manager.master_model.update()

    bc.config.row_generation.initialization_callback = add_constraint

if adaptive_cfg := config.get("adaptive_timeout"):
    bc.config.row_generation.subproblem_callback = adaptive_gurobi_timeout(
        initial_timeout=adaptive_cfg.get("initial", 1.0),
        final_timeout=adaptive_cfg.get("final", 30.0),
        transition_iterations=adaptive_cfg.get("transition_iterations", 15),
    )


if rank == 0:
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")

# result = bc.row_generation.solve()
# theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)
def strip_master_constraints(boot, rowgen):
    rowgen.strip_slack_constraints(50)
    bc.row_generation.strip_constraints_hard_threshold(30)


timeout_callback = adaptive_gurobi_timeout(
    initial_timeout=1.0,
    final_timeout=1.0,
    transition_iterations=10,
    strategy='linear',
    log=True
    )

se_result = bc.standard_errors.compute_bayesian_bootstrap(num_bootstrap=NUM_BOOTSTRAP, 
                                                            seed=SEED, 
                                                            verbose = True,
                                                            row_gen_iteration_callback = timeout_callback,
                                                            bootstrap_callback = strip_master_constraints
                                                            )

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
