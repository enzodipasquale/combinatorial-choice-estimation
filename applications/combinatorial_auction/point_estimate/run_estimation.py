#!/bin/env python
import sys, os, json, yaml
import numpy as np
from pathlib import Path
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
DELTA, WINNERS_ONLY, HQ_DISTANCE = app.get("delta", 4), app.get("winners_only", False), app.get("hq_distance", False)
ERROR_SEED = app.get("error_seed", 1995)
OUTPUT_DIR = APP_DIR / "estimation_results"

def get_input_dir():
    suffix = f"delta{DELTA}" + ("_winners" if WINNERS_ONLY else "") + ("_hqdist" if HQ_DISTANCE else "")
    return APP_DIR / "data/input_data" / suffix

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation"]})

if rank == 0:
    input_dir = get_input_dir()
    input_data = bc.data.load_quadratic_data_from_directory(input_dir)
    input_data["item_data"]["modular"] = -np.eye(bc.n_items)
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, features={bc.n_features}")
else:
    input_data = None

bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load()

feature_names = bc.config.dimensions.feature_names or []

if DELTA == 2 and feature_names:
    theta_lbs, theta_ubs = np.zeros(bc.n_features), np.full(bc.n_features, 1000.0)
    bounds_map = {"bidder_elig_pop": (75, 1000), "pop_distance": (400, 650), "travel_survey": (-120, 1000), "air_travel": (-75, 1000)}
    for i, name in enumerate(feature_names):
        if name in bounds_map:
            theta_lbs[i], theta_ubs[i] = bounds_map[name]
        elif name.startswith("FE_"):
            theta_lbs[i], theta_ubs[i] = 0, 1000
    bc.config.row_generation.theta_lbs = theta_lbs
    bc.config.row_generation.theta_ubs = theta_ubs
    if rank == 0:
        print("Custom bounds for delta=2 applied")

if adaptive_cfg := config.get("adaptive_timeout"):
    bc.config.row_generation.subproblem_callback = adaptive_gurobi_timeout(
        initial_timeout=adaptive_cfg.get("initial", 1.0),
        final_timeout=adaptive_cfg.get("final", 30.0),
        transition_iterations=adaptive_cfg.get("transition_iterations", 15),
    )

theta_warmstart = None
if app.get("use_previous_theta") and (theta_path := OUTPUT_DIR / "theta.npy").exists():
    theta_warmstart = np.load(theta_path) if rank == 0 else None
    theta_warmstart = comm.bcast(theta_warmstart, root=0)
    if rank == 0 and theta_warmstart.shape[0] != bc.n_features:
        theta_warmstart = None

result = bc.row_generation.solve(theta_warmstart=theta_warmstart)

if rank == 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save_npy(OUTPUT_DIR / "theta.npy")
    result.export_csv(
        OUTPUT_DIR / "theta_hat.csv",
        metadata={"delta": DELTA, "winners_only": WINNERS_ONLY, "hq_distance": HQ_DISTANCE,
                  "n_obs": bc.n_obs, "n_items": bc.n_items, "n_features": bc.n_features,
                  "n_simulations": bc.n_simulations, "num_mpi": comm.Get_size()},
        feature_names=feature_names,
    )
    from datetime import datetime
    json.dump({
        "delta": DELTA, "winners_only": WINNERS_ONLY, "hq_distance": HQ_DISTANCE,
        "n_features": bc.n_features, "feature_names": feature_names,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "converged": result.converged, "num_iterations": result.num_iterations,
    }, open(OUTPUT_DIR / "theta_metadata.json", "w"), indent=2)
    print(result.summary())
