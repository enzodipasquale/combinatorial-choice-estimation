#!/bin/env python
import sys, yaml, json
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

IS_LOCAL = Path("/Users/enzo-macbookpro").exists()
config = yaml.safe_load(open(BASE_DIR / ("config_local.yaml" if IS_LOCAL else "config.yaml")))

app = config.get("application", {})
DELTA, WINNERS_ONLY, HQ_DISTANCE = app.get("delta", 4), app.get("winners_only", False), app.get("hq_distance", False)
ERROR_SEED = app.get("error_seed", 1995)
OUTPUT_DIR = APP_DIR / "estimation_results"

bc = BundleChoice()
bc.load_config({k: v for k, v in config.items() if k in ["dimensions", "subproblem", "row_generation"]})

if rank == 0:
    input_data = prepare_data_main(delta=DELTA, winners_only=WINNERS_ONLY, hq_distance=HQ_DISTANCE, save_data=False)
    print(f"delta={DELTA}, agents={bc.n_obs}, items={bc.n_items}, features={bc.n_features}")
else:
    input_data = None

bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=ERROR_SEED)
bc.subproblems.load_subproblem()

if DELTA == 2:
    theta_lbs, theta_ubs = np.zeros(bc.n_features), np.full(bc.n_features, 1000.0)
    bounds_map = {"bidder_elig_pop": (75, 1000), "pop_distance": (400, 650), "travel_survey": (-120, 1000), "air_travel": (-75, 1000)}
    bc.config.row_generation.theta_lbs = theta_lbs
    bc.config.row_generation.theta_ubs = theta_ubs


if rank == 0:
    print(input_data.keys())
    print(input_data["id_data"].keys())
    print(input_data["item_data"].keys())

    print(input_data["id_data"]["modular"].shape)
    print(input_data["item_data"]["quadratic"].shape)

    print(bc.n_features)

result = bc.row_generation.solve(verbose=True)

# if rank == 0:
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     result.save_npy(OUTPUT_DIR / "theta.npy")
#     result.export_csv(
#         OUTPUT_DIR / "theta_hat.csv",
#         metadata={"delta": DELTA, "winners_only": WINNERS_ONLY, "hq_distance": HQ_DISTANCE,
#                   "n_obs": bc.n_obs, "n_items": bc.n_items, "n_features": bc.n_features,
#                   "n_simulations": bc.n_simulations, "num_mpi": comm.Get_size()},
#         feature_names=feature_names,
#     )
#     json.dump({
#         "delta": DELTA, "winners_only": WINNERS_ONLY, "hq_distance": HQ_DISTANCE,
#         "n_features": bc.n_features, "feature_names": feature_names,
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#         "converged": result.converged, "num_iterations": result.num_iterations,
#     }, open(OUTPUT_DIR / "theta_metadata.json", "w"), indent=2)
#     print(result.summary())
