#!/usr/bin/env python3
# MTA-level counterfactual allocation. Parameters rescaled by 1/α₁ into $.
# MTA item FE solved by row generation = MTA-level prices.

import sys, json, yaml
from pathlib import Path
import numpy as np
from mpi4py import MPI

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent.parent
PROJECT_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combchoice as cc
from combchoice.estimation.callbacks import adaptive_gurobi_timeout
from applications.combinatorial_auction.counterfactuals.MTA_licenses.prepare_data_counterfactual import (
    main as prepare_mta,
)
from applications.combinatorial_auction.results import load_result, save_counterfactual, OUTPUT_DIR

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Load configs ─────────────────────────────────────────────────────

cf_config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
est_cfg = cf_config["estimation"]

# ── Load estimation result from CSV ──────────────────────────────────

est = load_result(source=est_cfg["source"], run_idx=est_cfg["run_idx"])

theta_est = est["theta_hat"]
n_item_quad = len(est["quadratic_regressors"])

# ── Load FE decomposition ────────────────────────────────────────────

fe_path = cf_config.get("fe_decomposition_path",
                        str(OUTPUT_DIR / "fe_decomposition.json"))
with open(fe_path) as f:
    fe_decomp = json.load(f)

alpha_1 = fe_decomp["alpha_1"]

# Rescale all structural parameters into price units
theta_est_rescaled = theta_est / alpha_1

if rank == 0:
    print(f"Loaded {est_cfg['source']} run (idx={est_cfg['run_idx']}): "
          f"n_covariates={est['n_covariates']}, quad={est['quadratic_regressors']}")
    print(f"FE decomp: α₁={alpha_1:.2e}")
    print(f"Rescaled theta[0]={theta_est_rescaled[0]:.4f}, "
          f"theta[-4:]={theta_est_rescaled[-n_item_quad:].round(4)}")

# ── Prepare MTA data (with ξ̂) ─────────────────────────────────────────

if rank == 0:
    mta_data = prepare_mta(
        winners_only=est["winners_only"],
        continental_only=est["continental_only"],
        rescale_features=est["rescale_features"],
        modular_regressors=est["modular_regressors"],
        quadratic_regressors=est["quadratic_regressors"],
        fe_decomp=fe_decomp,
    )
    id_data = mta_data["id_data"]
    item_data = mta_data["item_data"]
    n_obs = id_data["obs_bundles"].shape[0]
    n_mod_agent = id_data["modular"].shape[-1]
    n_mod_item = item_data["modular"].shape[-1]      # n_mtas + n_item_quad + 1 (ξ)
else:
    mta_data = None
    n_obs, n_mod_agent, n_mod_item = 0, 0, 0

n_obs, n_mod_agent, n_mod_item = comm.bcast((n_obs, n_mod_agent, n_mod_item), root=0)

# MTA theta layout:
# [n_mod_agent agent | n_mtas FE | n_quad diag | 1 ξ coeff | n_quad offdiag]
n_extra = 1  # ξ̂ column
n_mtas = n_mod_item - n_item_quad - n_extra
n_covariates_mta = n_mod_agent + n_mod_item + n_item_quad

# ── Map BTA theta → MTA theta ───────────────────────────────────────

theta_init = np.zeros(n_covariates_mta)

# Agent modular: all rescaled agent modular coefficients from BTA estimation
theta_init[:n_mod_agent] = theta_est_rescaled[:n_mod_agent]

# BTA theta layout: [n_id_mod | n_items_bta FE | n_quad diag | n_quad offdiag]
n_items_bta = est["n_items"]
bta_quad_diag = theta_est_rescaled[n_mod_agent + n_items_bta : n_mod_agent + n_items_bta + n_item_quad]
bta_quad_offdiag = theta_est_rescaled[-n_item_quad:]

# Quadratic diagonals (from item_modular): rescaled BTA diagonal quad params
quad_diag_start = n_mod_agent + n_mtas
theta_init[quad_diag_start : quad_diag_start + n_item_quad] = bta_quad_diag

# ξ̂ coefficient: data stores (ξ̂-α₀)/α₁ per BTA, so coeff = 1.0
xi_idx = quad_diag_start + n_item_quad
theta_init[xi_idx] = 1.0

# Quadratic off-diagonals: rescaled BTA off-diagonal quad params
offdiag_start = xi_idx + 1
theta_init[offdiag_start:] = bta_quad_offdiag

if rank == 0:
    print(f"\nMTA: n_mtas={n_mtas}, n_covariates={n_covariates_mta}, "
          f"n_mod_item={n_mod_item}")
    print(f"theta_init non-zero positions: {np.where(theta_init != 0)[0]}")
    print(f"theta_init non-zero values: {theta_init[theta_init != 0].round(6)}")

# ── Setup Model ───────────────────────────────────────────────

auction = cc.Model()
cfg = {
    "dimensions": {
        "n_obs": n_obs,
        "n_items": n_mtas,
        "n_covariates": n_covariates_mta,
        "n_simulations": cf_config["dimensions"]["n_simulations"],
    },
    "subproblem": cf_config["subproblem"],
    "row_generation": {
        "max_iters": cf_config["row_generation"]["max_iters"],
        "tolerance": cf_config["row_generation"]["tolerance"],
        "parameters_to_log": list(range(n_mod_agent + n_mtas, n_covariates_mta)),
        "theta_bounds": cf_config["row_generation"]["theta_bounds"],
        "master_gurobi_params": cf_config["row_generation"].get("master_gurobi_params"),
    },
}
auction.load_config(cfg)
auction.data.load_and_distribute_input_data(mta_data if rank == 0 else None)
auction.features.build_quadratic_covariates_from_data()
auction.features.build_local_modular_error_oracle(seed=est["error_seed"])
auction.subproblems.load_solver()


def fix_theta(row_gen):
    # Warm-start and fix estimated params; MTA item FE are free (= prices)
    if rank != 0 or row_gen.master_model is None:
        return
    theta, _ = row_gen.master_variables
    for i in range(theta.size):
        theta[i].Start = theta_init[i]
    # Fix everything except the n_mtas item FE
    fixed = (list(range(n_mod_agent))
             + list(range(n_mod_agent + n_mtas, n_covariates_mta)))
    for i in fixed:
        theta[i].LB = theta[i].UB = theta_init[i]
    row_gen.master_model.update()
    row_gen.master_model.optimize()


pt_timeout_cb, _ = adaptive_gurobi_timeout(cf_config["callbacks"])

# ── Solve ────────────────────────────────────────────────────────────

result = auction.row_generation.solve(
    initialization_callback=fix_theta,
    iteration_callback=pt_timeout_cb,
    verbose=True,
)

# ── Save results ─────────────────────────────────────────────────────

if rank == 0 and result is not None:
    # With -I convention, utility from FE = -θ_FE[m], so θ_FE[m] = price_m
    mta_fe = result.theta_hat[n_mod_agent : n_mod_agent + n_mtas]
    mta_prices = mta_fe

    print(f"\nConverged: {result.converged}, iterations: {result.num_iterations}")
    print(f"MTA prices: min={mta_prices.min():.0f}, max={mta_prices.max():.0f}, "
          f"total={mta_prices.sum():.0f}")

    # Welfare from master LP u variables (in M$ since theta is rescaled)
    n_sim = cf_config["dimensions"]["n_simulations"]
    u_hat = result.u_hat  # (n_obs * n_sim,)
    u_per_obs = u_hat.reshape(n_sim, n_obs).mean(axis=0)  # avg over sims
    bidder_surplus = u_per_obs.sum()
    revenue = mta_prices.sum()
    total_surplus = bidder_surplus + revenue

    print(f"\nWelfare (M$):")
    print(f"  Bidder surplus (Σ u_i, avg over sims) = ${bidder_surplus:,.1f}M")
    print(f"  Revenue (Σ prices)                    = ${revenue:,.1f}M")
    print(f"  Total surplus                         = ${total_surplus:,.1f}M")

    save_counterfactual(cf_config, est, result, n_mtas)
