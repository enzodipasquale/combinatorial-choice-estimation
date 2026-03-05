#!/usr/bin/env python3
"""Joint estimation of C-block + A/B-block auctions.

Shared structural params, auction-specific FEs, item_mask enforcement.
Error oracle draws BTA-level normals; C-block uses directly, A/B aggregates via A^T.

Usage:  python run_estimation.py  |  mpirun -n 4 python run_estimation.py
"""

import json, sys, yaml
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent.parent.parent))

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
        quadratic_id_regressors=app.get("quadratic_id_regressors"),
    )
    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = input_data["item_data"]["modular"].shape[-1]
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    n_covariates = n_id_mod + n_item_mod + n_id_quad + n_item_quad
    n_btas, n_mtas = meta["n_btas"], meta["n_mtas"]
    n_obs_c, n_obs_ab = meta["n_obs_c"], meta["n_obs_ab"]
    A = meta["A"]

    config["dimensions"].update(n_obs=n_obs, n_items=n_items, n_covariates=n_covariates)
    config["application"]["n_obs_c"] = n_obs_c

    # Widen bounds for non-FE parameters
    bounds = config["row_generation"]["theta_bounds"]
    id_quad_off = n_id_mod + n_item_mod
    item_quad_off = id_quad_off + n_id_quad
    for k in list(range(1, n_id_mod)) + list(range(id_quad_off, id_quad_off + n_id_quad)) + list(range(item_quad_off, item_quad_off + n_item_quad)):
        bounds.setdefault("lbs", {})[k] = -1000
        bounds.setdefault("ubs", {})[k] = 1000

    print(f"\nJoint: {n_obs} obs ({n_obs_c}C+{n_obs_ab}AB), {n_items} items ({n_btas}BTA+{n_mtas}MTA), {n_covariates} cov")
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
    id_quad_off = n_id_mod + n_item_mod
    item_quad_off = id_quad_off + n_id_quad

    print(f"\ntheta ({len(theta)} params):")
    print(f"  elig_pop:  {theta[0]:.4f}")
    print(f"  BTA FE:    [{theta[n_id_mod:n_id_mod+n_btas].min():.4f}, {theta[n_id_mod:n_id_mod+n_btas].max():.4f}]")
    print(f"  MTA FE:    [{theta[n_id_mod+n_btas:n_id_mod+n_items].min():.4f}, {theta[n_id_mod+n_btas:n_id_mod+n_items].max():.4f}]")
    print(f"  id_quad:   {theta[id_quad_off:id_quad_off+n_id_quad]}")
    print(f"  item_quad: {theta[item_quad_off:item_quad_off+n_item_quad]}")

    out = {
        "theta_hat": theta.tolist(),
        "n_items": n_items, "n_obs": n_obs, "n_covariates": n_covariates,
        "n_id_mod": n_id_mod, "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad, "n_item_quad": n_item_quad,
        "n_btas": n_btas, "n_mtas": n_mtas,
        "n_obs_c": n_obs_c, "n_obs_ab": n_obs_ab,
        "specification": {k: app.get(f"{k}_regressors", []) for k in ["modular", "quadratic", "quadratic_id"]},
        "continental_mta_nums": [int(m) for m in meta["continental_mta_nums"]],
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
    }
    out_path = BASE_DIR / "joint_estimation_result.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")
