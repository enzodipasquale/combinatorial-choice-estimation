#!/usr/bin/env python3
"""C-block estimation with price instead of item FEs.

Same as compute_point_estimate.py but replaces -I (480 FEs) with
-price (1 covariate) in item_data["modular"].

Usage:  python compute_point_estimate_price.py  |  mpirun -n 4 python compute_point_estimate_price.py
"""

import json, sys, yaml
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.data.prepare_data import (
    main as prepare_data_main, load_raw_data, build_context,
)

# ── MPI ──────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

# ── Config (use joint config for matching specification) ─────────────
JOINT_DIR = APP_DIR / "joint"
config = yaml.safe_load(open(JOINT_DIR / "config.yaml"))
app = config.get("application", {})

# ── Data (rank 0 only) ──────────────────────────────────────────────
if rank == 0:
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

    input_data = prepare_data_main(
        winners_only=False,
        continental_only=True,
        modular_regressors=app.get("modular_regressors"),
        quadratic_regressors=app.get("quadratic_regressors"),
        quadratic_id_regressors=app.get("quadratic_id_regressors", []),
    )

    # ── Replace FEs with price ────────────────────────────────────
    raw = load_raw_data(continental_only=True)
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    input_data["item_data"]["modular"] = -price_bta[:, None]  # (n_items, 1)

    print(f"  Price (B$): [{price_bta.min():.4f}, {price_bta.max():.4f}]")

    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = 1  # price
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_covariates = n_id_mod + n_item_mod + n_id_quad + n_item_quad

    modular_regressors = app.get("modular_regressors", [])
    quadratic_id_regressors = app.get("quadratic_id_regressors", [])
    quadratic_regressors = app.get("quadratic_regressors", [])

    # Named covariates
    covariate_names = {i: name for i, name in enumerate(modular_regressors)}
    covariate_names[n_id_mod] = "price"
    id_quad_off = n_id_mod + n_item_mod
    for i, name in enumerate(quadratic_id_regressors):
        covariate_names[id_quad_off + i] = name
    item_quad_off = id_quad_off + n_id_quad
    for i, name in enumerate(quadratic_regressors):
        covariate_names[item_quad_off + i] = name

    config["dimensions"].update(n_obs=n_obs, n_items=n_items, n_covariates=n_covariates,
                                covariate_names=covariate_names)

    # All params are structural — widen bounds
    bounds = config["row_generation"]["theta_bounds"]
    for k in range(n_covariates):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000

    print(f"\nC-block (price): {n_obs} obs, {n_items} items, {n_covariates} cov")
    print(f"  Covariates: {covariate_names}")
else:
    input_data = None

if comm is not None:
    config = comm.bcast(config, root=0)

# ── Model ────────────────────────────────────────────────────────────
model = ce.Model()
model.load_config(config)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()

# ── Error oracle ─────────────────────────────────────────────────────
seed = app.get("error_seed", 1998)
model.features.build_local_modular_error_oracle(seed=seed)

# ── Solve ────────────────────────────────────────────────────────────
model.subproblems.load_solver()
pt_cb, _ = adaptive_gurobi_timeout(config.get("callbacks", {})["row_gen"])
result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)

# ── Save ─────────────────────────────────────────────────────────────
if rank == 0 and result is not None:
    theta = result.theta_hat

    print(f"\ntheta ({len(theta)} params):")
    for idx, name in sorted(covariate_names.items()):
        print(f"  {name}: {theta[idx]:.4f}")

    out = {
        "theta_hat": theta.tolist(),
        "n_items": n_items, "n_obs": n_obs, "n_covariates": n_covariates,
        "n_id_mod": n_id_mod, "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad, "n_item_quad": n_item_quad,
        "use_price": True,
        "specification": {k: app.get(f"{k}_regressors", []) for k in ["modular", "quadratic", "quadratic_id"]},
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
    }
    out_path = BASE_DIR / "bta_price_estimation_result.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")
