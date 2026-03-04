#!/usr/bin/env python3
"""Cross-auction regression: identify price coefficient α.

Compares C-block BTA FE (aggregated to MTA) with A/B block MTA FE.
Both estimations must use the same specification so FE residualize equivalently.

    Y_m = γ₀ + γ₁·|m| + α·X_m + ε_m

    Y_m = Σ_{j∈m} θ̂^C_FE[j] − δ̂^{AB}_m
    X_m = Σ_{j∈m} p^C_j − p^{AB}_m
    |m| = number of BTAs in MTA m
"""

import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.prepare_data import load_raw_data, build_context, build_features, QUADRATIC
from applications.combinatorial_auction.counterfactuals.MTA_licenses.prepare_data_counterfactual import (
    load_aggregation_matrix,
)

AB_DIR = APP_DIR / "ab_block"
BTA_DIR = APP_DIR / "point_estimate"


def load_results():
    bta = json.load(open(BTA_DIR / "bta_estimation_result.json"))
    ab = json.load(open(AB_DIR / "ab_estimation_result.json"))
    return bta, ab


def extract_fe(result):
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_items = result["n_items"]
    return theta[n_id : n_id + n_items]


def load_prices(raw, continental_mta_nums, A):
    # BTA prices (C-block winning bids, raw dollars)
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float)

    # A/B prices (average of A and B block winning bids per MTA)
    winners = pd.read_csv(AB_DIR / "data" / "winning_bids.csv")
    winners = winners[winners["mta_num"].isin(continental_mta_nums)]
    mta_avg = winners.groupby("mta_num")["price"].mean()
    price_ab = np.array([mta_avg.get(m, 0.0) for m in continental_mta_nums])

    # aggregate BTA prices to MTA
    price_mta_c = A @ price_bta
    return price_mta_c, price_ab, price_bta


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def run(include_diag_quad=False, quadratic_regressors=None):
    bta_res, ab_res = load_results()
    fe_bta = extract_fe(bta_res)
    fe_ab = extract_fe(ab_res)

    raw = load_raw_data(continental_only=True)
    continental_btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(continental_btas)
    n_mtas = A.shape[0]

    continental_mta_nums = ab_res["continental_mta_nums"]
    price_mta_c, price_ab, price_bta = load_prices(raw, continental_mta_nums, A)

    # aggregate BTA FE to MTA
    fe_mta_c = A @ fe_bta
    n_btas_per_mta = A.sum(axis=1)

    Y = fe_mta_c - fe_ab
    X_price = price_mta_c - price_ab
    X = np.column_stack([np.ones(n_mtas), n_btas_per_mta, X_price])
    col_names = ["const", "|m|", "α (price)"]

    # within-MTA complementarity controls
    if include_diag_quad:
        quad_names = quadratic_regressors or ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]
        ctx = build_context(raw)
        bta_quad = build_features(QUADRATIC, quad_names, ctx)
        n_qfeat = bta_quad.shape[-1]
        Q_mta = np.stack([A @ bta_quad[:, :, k] @ A.T for k in range(n_qfeat)], axis=-1)
        diag_quad = np.array([np.diag(Q_mta[:, :, k]) for k in range(n_qfeat)]).T
        X = np.column_stack([X, diag_quad])
        col_names += [f"diag_{name}" for name in quad_names]

    beta, se, r2, resid = ols(X, Y)

    print(f"\n{'='*60}")
    print(f"Cross-auction regression  (N = {n_mtas} MTAs)")
    print(f"{'='*60}")
    print(f"  BTA result: {bta_res['n_items']} items, {bta_res['n_id_mod']} id_mod, converged={bta_res['converged']}")
    print(f"  A/B result: {ab_res['n_items']} items, {ab_res['n_id_mod']} id_mod, converged={ab_res['converged']}")
    print(f"\n  {'Covariate':<20} {'Estimate':>12} {'SE':>12} {'t-stat':>10}")
    print(f"  {'-'*54}")
    for name, b, s in zip(col_names, beta, se):
        print(f"  {name:<20} {b:>12.6f} {s:>12.6f} {b/s:>10.3f}")
    print(f"\n  R² = {r2:.4f}")
    print(f"  Residual std = {resid.std():.4f}")

    return {"beta": beta, "se": se, "r2": r2, "col_names": col_names, "resid": resid}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("\n── Baseline (no complementarity controls) ──")
    run(include_diag_quad=False)

    print("\n── With within-MTA complementarity controls ──")
    run(include_diag_quad=True)
