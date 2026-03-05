#!/usr/bin/env python3
"""Estimate max-score model on A/B block (Auction 4) data.

46 continental MTAs as items (each MTA has 2 winners from blocks A and B).
30 bidders.  One FE per MTA (46 parameters).

Usage:
    python run_estimation.py              # single-process
    mpirun -n 4 python run_estimation.py  # parallel
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
from applications.combinatorial_auction.ab_block.prepare_data import main as prepare_ab_data

# ── MPI (optional) ────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    comm = None
    rank = 0

# ── Config ────────────────────────────────────────────────────────────
config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config.get("application", {})

# ── Data ──────────────────────────────────────────────────────────────
if rank == 0:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    input_data, meta = prepare_ab_data(
        modular_regressors=app.get("modular_regressors"),
        quadratic_regressors=app.get("quadratic_regressors"),
        quadratic_id_regressors=app.get("quadratic_id_regressors"),
    )

    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = input_data["item_data"]["modular"].shape[-1]
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_covariates = n_id_mod + n_item_mod + n_id_quad + n_item_quad

    config["dimensions"].update({
        "n_obs": n_obs, "n_items": n_items, "n_covariates": n_covariates,
    })

    # bounds for id modular (all except first)
    bounds = config["row_generation"]["theta_bounds"]
    for k in range(1, n_id_mod):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000
    # bounds for id quadratic
    id_quad_offset = n_id_mod + n_item_mod
    for k in range(id_quad_offset, id_quad_offset + n_id_quad):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000
    # bounds for item quadratic
    item_quad_offset = id_quad_offset + n_id_quad
    for k in range(item_quad_offset, item_quad_offset + n_item_quad):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000

    A = meta["A"]

    print(f"\nA/B estimation: {n_obs} agents, {n_items} items (MTAs), {n_covariates} covariates")
    print(f"  id_mod={n_id_mod}, item_mod={n_item_mod} ({n_items} FE + "
          f"{n_item_mod - n_items} diag), id_quad={n_id_quad}, item_quad={n_item_quad}")
else:
    input_data = None
    A = None

if comm is not None:
    config = comm.bcast(config, root=0)
    A = comm.bcast(A, root=0)

# ── Model ─────────────────────────────────────────────────────────────
auction = ce.Model()
auction.load_config(config)
auction.data.load_and_distribute_input_data(input_data)
auction.features.build_quadratic_covariates_from_data()
# Build BTA-level errors aggregated to MTA (consistent with C-block error structure)
seed = app.get("error_seed", 1998)
n_btas = A.shape[1]
n_local = auction.features.comm_manager.num_local_agent
bta_errors = np.zeros((n_local, n_btas))
for i, global_id in enumerate(auction.features.comm_manager.agent_ids):
    rng = np.random.default_rng((seed, global_id))
    bta_errors[i] = rng.normal(0, 1, n_btas)
auction.features.local_modular_errors = bta_errors @ A.T
auction.features._error_oracle = lambda bundles, ids: (auction.features.local_modular_errors[ids] * bundles).sum(-1)
auction.features._error_oracle_vectorized = True
auction.features._error_oracle_takes_data = False
auction.subproblems.load_solver()

callbacks = config.get("callbacks", {})
pt_timeout_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])

# ── Solve ─────────────────────────────────────────────────────────────
result = auction.row_generation.solve(
    iteration_callback=pt_timeout_cb,
    verbose=True,
)

# ── Save ──────────────────────────────────────────────────────────────
if rank == 0 and result is not None:
    theta = result.theta_hat
    print(f"\ntheta_hat ({len(theta)} params):")
    print(f"  id_mod:    {theta[:n_id_mod]}")
    print(f"  item FE:   [{theta[n_id_mod:n_id_mod+n_items].min():.4f}, "
          f"{theta[n_id_mod:n_id_mod+n_items].max():.4f}]")

    out = {
        "theta_hat": theta.tolist(),
        "n_items": n_items,
        "n_obs": n_obs,
        "n_covariates": n_covariates,
        "n_id_mod": n_id_mod,
        "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad,
        "n_item_quad": n_item_quad,
        "specification": {
            "modular": app.get("modular_regressors", []),
            "quadratic": app.get("quadratic_regressors", []),
            "quadratic_id": app.get("quadratic_id_regressors", []),
        },
        "n_mtas": meta["n_mtas"],
        "continental_mta_nums": [int(m) for m in meta["continental_mta_nums"]],
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
    }
    out_path = BASE_DIR / "ab_estimation_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
