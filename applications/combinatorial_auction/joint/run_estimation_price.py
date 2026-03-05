#!/usr/bin/env python3
"""Joint estimation with price covariate instead of item FEs.

Uses prepare_data to build the standard joint data, then replaces
item_data["modular"] = -I (FEs) with a single price column (B$).
Estimates α directly in the structural model.

Usage:  python run_estimation_price.py  |  mpirun -n 4 python run_estimation_price.py
"""

import json, sys, yaml
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.joint.prepare_data import main as prepare_joint_data

# ── MPI ──────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

# ── Config ───────────────────────────────────────────────────────────
config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config.get("application", {})

# ── Data (rank 0 only) ──────────────────────────────────────────────
if rank == 0:
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

    input_data, meta = prepare_joint_data(
        modular_regressors=app.get("modular_regressors"),
        quadratic_regressors=app.get("quadratic_regressors"),
        quadratic_id_regressors=app.get("quadratic_id_regressors", []),
        ab_elig_bandwidth=app.get("ab_elig_bandwidth", 30.0),
    )
    n_btas, n_mtas = meta["n_btas"], meta["n_mtas"]
    n_items = n_btas + n_mtas
    mta_nums = meta["continental_mta_nums"]

    # ── Replace FEs with price ───────────────────────────────────────
    from applications.combinatorial_auction.data.prepare_data import load_raw_data
    raw = load_raw_data(continental_only=True)

    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    ab_winners = pd.read_csv(APP_DIR / "block_ab" / "data" / "winning_bids.csv")
    ab_winners = ab_winners[ab_winners["mta_num"].isin(mta_nums)]
    mta_avg = ab_winners.groupby("mta_num")["price"].mean()
    price_mta = np.array([mta_avg.get(m, 0.0) for m in mta_nums]) / 1e9

    item_prices = np.concatenate([price_bta, price_mta])
    input_data["item_data"]["modular"] = -item_prices[:, None]  # (n_items, 1)

    print(f"  Price (B$): BTA [{price_bta.min():.4f}, {price_bta.max():.4f}], "
          f"MTA [{price_mta.min():.4f}, {price_mta.max():.4f}]")

    n_obs = input_data["id_data"]["obs_bundles"].shape[0]
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = 1  # price
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_covariates = n_id_mod + n_item_mod + n_id_quad + n_item_quad
    n_obs_c, n_obs_ab = meta["n_obs_c"], meta["n_obs_ab"]
    A = meta["A"]

    # Named covariates — all structural, no FEs
    id_quad_off = n_id_mod + n_item_mod
    item_quad_off = id_quad_off + n_id_quad
    covariate_names = {i: name for i, name in enumerate(app.get("modular_regressors", []))}
    covariate_names[n_id_mod] = "price"
    for i, name in enumerate(app.get("quadratic_id_regressors", [])):
        covariate_names[id_quad_off + i] = name
    for i, name in enumerate(app.get("quadratic_regressors", [])):
        covariate_names[item_quad_off + i] = name

    config["dimensions"].update(n_obs=n_obs, n_items=n_items, n_covariates=n_covariates,
                                covariate_names=covariate_names)
    config["application"]["n_obs_c"] = n_obs_c

    # All params are structural — widen bounds
    bounds = config["row_generation"]["theta_bounds"]
    for k in range(n_covariates):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000

    print(f"\nJoint (price): {n_obs} obs ({n_obs_c}C+{n_obs_ab}AB), "
          f"{n_items} items ({n_btas}BTA+{n_mtas}MTA), {n_covariates} cov")
    print(f"  Covariates: {covariate_names}")
else:
    input_data, A = None, None

if comm is not None:
    config = comm.bcast(config, root=0)
    A = comm.bcast(A, root=0)

# ── Model ────────────────────────────────────────────────────────────
model = ce.Model()
model.load_config(config)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()

# ── Error oracle: BTA draws, auction-specific routing ────────────────
seed = app.get("error_seed", 1998)
n_bta = A.shape[1]
n_obs_c_all = config["application"]["n_obs_c"]
cm = model.features.comm_manager

local_errors = np.zeros((cm.num_local_agent, model.n_items))
for i, gid in enumerate(cm.agent_ids):
    rng = np.random.default_rng((seed, gid))
    bta_err = rng.normal(0, 1, n_bta)
    if gid % model.n_obs < n_obs_c_all:
        local_errors[i, :n_bta] = bta_err
    else:
        local_errors[i, n_bta:] = bta_err @ A.T

model.features.local_modular_errors = local_errors
model.features._error_oracle = lambda bundles, ids: (model.features.local_modular_errors[ids] * bundles).sum(-1)
model.features._error_oracle_vectorized = True
model.features._error_oracle_takes_data = False

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
        "n_btas": n_btas, "n_mtas": n_mtas,
        "n_obs_c": n_obs_c, "n_obs_ab": n_obs_ab,
        "use_price": True,
        "specification": {k: app.get(f"{k}_regressors", []) for k in ["modular", "quadratic", "quadratic_id"]},
        "continental_mta_nums": [int(m) for m in meta["continental_mta_nums"]],
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
    }
    out_path = BASE_DIR / "joint_price_result.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")
