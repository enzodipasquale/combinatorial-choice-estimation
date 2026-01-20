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

def get_input_dir():
    suffix = f"delta{DELTA}"
    if app.get("winners_only"): suffix += "_winners"
    if app.get("hq_distance"): suffix += "_hqdist"
    return APP_DIR / "data/input_data" / suffix

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation", "standard_errors"]})

if rank == 0:
    input_dir = get_input_dir()
    input_data = bc.data.load_quadratic_data_from_directory(
        input_dir,
        additional_agent_data={
            "capacity": str(input_dir / "constraints" / "capacity.csv"),
            "obs_bundles": str(input_dir / "obs_bundles.csv"),
        },
        additional_item_data={
            "weights": str(input_dir / "constraints" / "weights.csv"),
        },
    )
    input_data["item_data"]["modular"] = -np.eye(bc.n_items)
else:
    input_data = None

bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load()

if theta_bounds := config.get("theta_bounds"):
    theta_lbs = np.zeros(bc.n_features)
    if "air_travel_lb" in theta_bounds:
        theta_lbs[-1] = theta_bounds["air_travel_lb"]
    if "travel_survey_lb" in theta_bounds:
        theta_lbs[-2] = theta_bounds["travel_survey_lb"]
    bc.config.row_generation.theta_lbs = theta_lbs

if config.get("constraints", {}).get("pop_dominates_travel"):
    def add_constraint(model, theta, u):
        model.addConstr(theta[-3] + theta[-2] + theta[-1] >= 0, "pop_dominates_travel")
    bc.config.row_generation.master_init_callback = add_constraint

if adaptive_cfg := config.get("adaptive_timeout"):
    bc.config.row_generation.subproblem_callback = adaptive_gurobi_timeout(
        initial_timeout=adaptive_cfg.get("initial", 1.0),
        final_timeout=adaptive_cfg.get("final", 30.0),
        transition_iterations=adaptive_cfg.get("transition_iterations", 15),
    )

feature_names = bc.config.dimensions.feature_names or []
structural_idx = [i for i, n in enumerate(feature_names) if not n.startswith("FE_")]

if rank == 0:
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, bootstrap={NUM_BOOTSTRAP}")

result = bc.row_generation.solve()
theta_hat = comm.bcast(result.theta_hat if rank == 0 else None, root=0)

se_result = bc.standard_errors.compute_bayesian_bootstrap(num_bootstrap=NUM_BOOTSTRAP, seed=SEED)

if rank == 0 and se_result is not None:
    theta_point = theta_hat
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "se_bootstrap.npz", mean=se_result.mean, se=se_result.se, 
             ci_lower=se_result.ci_lower, ci_upper=se_result.ci_upper)
    
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "delta": DELTA,
           "winners_only": app.get("winners_only"), "hq_distance": app.get("hq_distance"),
           "n_obs": bc.n_obs, "n_items": bc.n_items, "n_features": bc.n_features,
           "num_bootstrap": NUM_BOOTSTRAP, "seed": SEED,
           "converged": result.converged, "num_iterations": result.num_iterations}
    for i in structural_idx:
        name = feature_names[i]
        row[f"theta_{name}"] = theta_point[i]
        row[f"se_{name}"] = se_result.se[i]
        row[f"t_{name}"] = se_result.t_stats[i]
    
    csv_path = OUTPUT_DIR / "se_bootstrap.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    
    print("\nResults:")
    for i in structural_idx:
        print(f"  {feature_names[i]}: θ={theta_point[i]:.4f}, SE={se_result.se[i]:.4f}")

comm.Barrier()
