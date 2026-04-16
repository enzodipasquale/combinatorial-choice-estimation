#!/usr/bin/env python3
"""Consistency check: verify estimation and counterfactual use the same data.

Loads data exactly as estimate.py and counterfactual/prepare.py + run.py do,
then checks that covariates and errors match after aggregation to MTA level.
"""
import sys, json, yaml
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent
REPO_ROOT = CBLOCK_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import combest as ce
from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import (
    load_bta_data, build_context, load_aggregation_matrix, build_cholesky_factor,
)
from applications.combinatorial_auction.scripts.c_block.counterfactual.prepare import (
    prepare_counterfactual,
)
from applications.combinatorial_auction.scripts.c_block.counterfactual.errors import (
    build_counterfactual_errors,
)


def check(config_path, cf_config_path):
    config = yaml.safe_load(open(config_path))
    cf_config = yaml.safe_load(open(cf_config_path))
    app = config["application"]
    cf_app = cf_config["application"]

    mod_names = app.get("modular_regressors", ["elig_pop"])
    quad_names = app.get("quadratic_regressors", [])
    qid_names = app.get("quadratic_id_regressors", [])
    error_scaling = app.get("error_scaling")
    error_seed = app.get("error_seed", 2006)

    print("=" * 70)
    print("CONSISTENCY CHECK: estimation vs counterfactual")
    print("=" * 70)
    print(f"  Estimation config: {config_path}")
    print(f"  CF config:         {cf_config_path}")
    print(f"  error_scaling:     {error_scaling}")
    print()

    # ── 1. Load data as estimation does ──────────────────────────────────
    raw = load_bta_data()
    ctx = build_context(raw)

    input_data, meta = prepare(
        dataset=app["dataset"],
        modular_regressors=mod_names,
        quadratic_regressors=quad_names,
        quadratic_id_regressors=qid_names,
        item_modular=app.get("item_modular", "fe"),
        n_simulations=config["dimensions"].get("n_simulations"),
    )
    meta.pop("raw", None)

    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    app.update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )

    # Build estimation model (no solving)
    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    # Build estimation errors exactly as estimate.py does
    L = build_cholesky_factor(app.get("error_correlation"))
    cov = L @ L.T if L is not None else None
    model.features.build_local_modular_error_oracle(seed=error_seed, covariance_matrix=cov)
    if error_scaling == "pop":
        pop_est = model.data.local_data.item_data['weight'].astype(float)
        pop_est = pop_est / pop_est.sum()
        model.features.local_modular_errors *= pop_est[None, :]

    est_errors = model.features.local_modular_errors.copy()
    est_id_mod = input_data["id_data"]["modular"].copy()
    est_item_mod = input_data["item_data"]["modular"].copy()
    est_capacity = input_data["id_data"]["capacity"].copy()
    est_obs = input_data["id_data"]["obs_bundles"].copy()
    est_weight = input_data["item_data"]["weight"].copy()

    n_obs = meta["n_obs"]
    n_btas = meta["n_items"]

    if "quadratic" in input_data["item_data"]:
        est_item_quad = input_data["item_data"]["quadratic"].copy()
    else:
        est_item_quad = None
    if "quadratic" in input_data["id_data"]:
        est_id_quad = input_data["id_data"]["quadratic"].copy()
    else:
        est_id_quad = None

    print(f"  Estimation: {n_obs} obs, {n_btas} items")

    # ── 2. Load data as counterfactual does ──────────────────────────────
    est_result_path = CBLOCK_DIR / "results" / cf_app["est_result"]
    cf_input, cf_meta = prepare_counterfactual(
        est_result_path_or_dict=str(est_result_path),
        alpha_0=cf_app["alpha_0"],
        alpha_1=cf_app["alpha_1"],
        demand_controls=cf_app.get("demand_controls"),
    )

    A = cf_meta["A"]
    n_mtas = cf_meta["n_mtas"]

    # Build CF errors exactly as run.py does
    offset = cf_meta["offset_m_no_xi"]
    L_corr = build_cholesky_factor(cf_app.get("error_correlation"))
    pop_cf = None
    if cf_app.get("error_scaling") == "pop":
        w = cf_meta["bta_weight"]
        pop_cf = w / w.sum()

    # To compare errors, build them for the same seed
    cf_seed = cf_app.get("error_seed", 24)

    print(f"  CF: {cf_meta['n_obs']} obs, {n_mtas} MTAs")
    print()

    # ── 3. Check covariates ──────────────────────────────────────────────
    ok = True

    # 3a. Modular: CF aggregates BTA modular via einsum('ijk,mj->imk', bta_mod, A)
    cf_id_mod = cf_input["id_data"]["modular"]  # (n_obs, n_mtas, n_id_mod)
    est_mod_agg = np.einsum('ijk,mj->imk', est_id_mod, A)
    diff = np.abs(cf_id_mod - est_mod_agg).max()
    status = "OK" if diff < 1e-10 else f"FAIL (max diff={diff:.2e})"
    print(f"  Modular covariates (aggregated):     {status}")
    if diff >= 1e-10:
        ok = False

    # 3b. Quadratic item: CF uses _aggregate_quadratics
    if est_item_quad is not None:
        cf_item_quad = cf_input["item_data"]["quadratic"]  # (n_mtas, n_mtas, n_quad)
        for k in range(est_item_quad.shape[-1]):
            Q_bta = est_item_quad[:, :, k]
            Q_mta_manual = A @ Q_bta @ A.T
            diff_q = np.abs(cf_item_quad[:, :, k] - Q_mta_manual).max()
            status_q = "OK" if diff_q < 1e-10 else f"FAIL (max diff={diff_q:.2e})"
            print(f"  Quadratic item [{k}] (A @ Q @ A.T):  {status_q}")
            if diff_q >= 1e-10:
                ok = False

    # 3c. Quadratic id: CF uses elig_i * (A @ Q @ A.T)
    if est_id_quad is not None:
        cf_id_quad = cf_input["id_data"]["quadratic"]  # (n_obs, n_mtas, n_mtas, n_id_quad)
        elig = ctx["elig"]
        for k in range(est_id_quad.shape[-1]):
            # est_id_quad[:, :, :, k] is (n_obs, n_btas, n_btas)
            # Should aggregate to elig_i * (A @ Q_item @ A.T)
            Q_bta_k = est_item_quad[:, :, k]  # use item quad as base
            Q_mta_k = A @ Q_bta_k @ A.T
            manual = elig[:, None, None] * Q_mta_k[None, :, :]
            diff_qid = np.abs(cf_id_quad[:, :, :, k] - manual).max()
            status_qid = "OK" if diff_qid < 1e-10 else f"FAIL (max diff={diff_qid:.2e})"
            print(f"  Quadratic id [{k}] (elig * A@Q@A.T): {status_qid}")
            if diff_qid >= 1e-10:
                ok = False

    # 3d. obs_bundles: CF aggregates to MTA level
    cf_obs = cf_input["id_data"]["obs_bundles"]
    est_obs_agg = (est_obs.astype(float) @ A.T > 0).astype(int)
    diff_obs = np.abs(cf_obs - est_obs_agg).max()
    status_obs = "OK" if diff_obs == 0 else f"FAIL (max diff={diff_obs})"
    print(f"  Observed bundles (aggregated):       {status_obs}")
    if diff_obs != 0:
        ok = False

    # 3e. Capacity: should be identical
    cf_cap = cf_input["id_data"]["capacity"]
    diff_cap = np.abs(cf_cap - est_capacity).max()
    status_cap = "OK" if diff_cap == 0 else f"FAIL (max diff={diff_cap})"
    print(f"  Capacity:                            {status_cap}")
    if diff_cap != 0:
        ok = False

    # 3f. Weight: CF aggregates BTA weight to MTA
    cf_weight = cf_input["item_data"]["weight"]
    est_weight_agg = (A @ est_weight.astype(float)).astype(int)
    diff_w = np.abs(cf_weight - est_weight_agg).max()
    status_w = "OK" if diff_w == 0 else f"FAIL (max diff={diff_w})"
    print(f"  Weight (aggregated):                 {status_w}")
    if diff_w != 0:
        ok = False

    # ── 4. Check pop scaling in errors ───────────────────────────────────
    print()
    if error_scaling == "pop":
        # Estimation pop: item_data['weight'] / sum
        pop_est_check = est_weight.astype(float)
        pop_est_check = pop_est_check / pop_est_check.sum()

        # CF pop: meta['bta_weight'] / sum
        pop_cf_check = cf_meta["bta_weight"]
        pop_cf_check = pop_cf_check / pop_cf_check.sum()

        diff_pop = np.abs(pop_est_check - pop_cf_check).max()
        status_pop = "OK" if diff_pop < 1e-12 else f"FAIL (max diff={diff_pop:.2e})"
        print(f"  Pop scaling vector:                  {status_pop}")
        if diff_pop >= 1e-12:
            ok = False
    else:
        print(f"  Pop scaling: N/A (error_scaling={error_scaling})")

    # ── 5. Check error structure ─────────────────────────────────────────
    # Build CF errors for agent 0 and compare aggregated estimation errors
    # Use estimation seed to make them comparable
    print()
    print("  Error structure check (seed alignment):")

    # Estimation errors for agent 0: est_errors[0, :] shape (n_btas,)
    # This was built with seed=(error_seed, agent_global_id)
    # CF errors use seed=(cf_seed, agent_global_id)
    # Seeds differ, so raw draws differ. But we can check the STRUCTURE:
    # - estimation: err_j = pop_j * normal_j (for pop scaling)
    # - CF: err_m = sum_{j in m} pop_j * normal_j + offset_m

    # What we CAN check: if we build CF errors with the ESTIMATION seed
    # and zero offset, the aggregated BTA errors should match
    zero_offset = np.zeros(n_mtas)
    cf_errors_check = build_counterfactual_errors(
        model.features.comm_manager, n_btas, A, zero_offset, error_seed,
        elig=cf_meta.get("elig"), error_scaling=error_scaling, L_corr=L_corr,
        pop=pop_cf,
    )

    # Estimation errors aggregated to MTA: est_errors @ A.T
    est_errors_mta = est_errors @ A.T

    diff_err = np.abs(cf_errors_check - est_errors_mta).max()
    status_err = "OK" if diff_err < 1e-10 else f"FAIL (max diff={diff_err:.2e})"
    print(f"  Errors (CF w/ zero offset vs est@A.T): {status_err}")
    if diff_err >= 1e-10:
        ok = False
        # Debug: check agent 0
        print(f"    Agent 0 est_errors@A.T[:5]: {est_errors_mta[0, :5]}")
        print(f"    Agent 0 cf_errors[:5]:      {cf_errors_check[0, :5]}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    if ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  *** SOME CHECKS FAILED ***")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--est-config", default=str(CBLOCK_DIR / "configs" / "boot_pop_scaling.yaml"))
    parser.add_argument("--cf-config", default=str(CBLOCK_DIR / "configs" / "cf_pop_scaling_blp.yaml"))
    args = parser.parse_args()
    check(args.est_config, args.cf_config)
