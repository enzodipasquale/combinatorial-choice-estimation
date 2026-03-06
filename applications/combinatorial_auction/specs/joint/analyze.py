#!/usr/bin/env python3
import json, sys
import numpy as np
from pathlib import Path

SPECS_DIR = Path(__file__).parent.parent
APP_DIR = SPECS_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import (
    load_bta_data, load_aggregation_matrix, load_ab_data,
)

JOINT_DIR = Path(__file__).parent


def extract_joint_fe(result):
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_btas, n_mtas = result["n_btas"], result["n_mtas"]
    return theta[n_id : n_id + n_btas], theta[n_id + n_btas : n_id + n_btas + n_mtas]


def load_prices(raw, mta_nums, A):
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float)
    winners, _ = load_ab_data()
    winners = winners[winners["mta_num"].isin(mta_nums)]
    mta_avg = winners.groupby("mta_num")["price"].mean()
    price_ab = np.array([mta_avg.get(m, 0.0) for m in mta_nums])
    return A @ price_bta / 1e9, price_ab / 1e9, price_bta / 1e9


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def _print_regression(title, n_mtas, col_names, beta, se, r2, resid):
    print(f"\n{'='*60}")
    print(f"{title}  (N = {n_mtas} MTAs)")
    print(f"{'='*60}")
    print(f"\n  {'Covariate':<20} {'Estimate':>12} {'SE':>12} {'t-stat':>10}")
    print(f"  {'-'*54}")
    for name, b, s in zip(col_names, beta, se):
        print(f"  {name:<20} {b:>12.6f} {s:>12.6f} {b/s:>10.3f}")
    print(f"\n  R2 = {r2:.4f}")
    print(f"  Residual std = {resid.std():.4f}")


def run_joint(result_file="result.json"):
    joint = json.load(open(JOINT_DIR / result_file))
    fe_bta, fe_ab = extract_joint_fe(joint)

    raw = load_bta_data()
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_mtas = A.shape[0]
    mta_nums = joint["continental_mta_nums"]
    price_mta_c, price_ab, price_bta = load_prices(raw, mta_nums, A)
    n_btas = len(fe_bta)

    # delta = -FE  (item_modular = -I, so utility contribution = -theta = delta)
    delta_bta = -fe_bta
    delta_ab = -fe_ab

    # (1) A@delta_BTA - delta_MTA ~ const + |m| - alpha_1*(price_c - price_ab)
    #     Under xi_m = sum xi_j, residual vanishes and coeff on price_diff = -alpha_1
    Y1 = A @ delta_bta - delta_ab
    X1 = np.column_stack([np.ones(n_mtas), A.sum(1), price_mta_c - price_ab])
    beta1, se1, r2_1, resid1 = ols(X1, Y1)
    _print_regression("A@delta_BTA - delta_MTA ~ const + |m| + (price_c - price_ab)",
                      n_mtas, ["const", "|m|", "-alpha_1"], beta1, se1, r2_1, resid1)

    # (2) delta_MTA ~ const + |m| + alpha_1*(-price_MTA)
    X2 = np.column_stack([np.ones(n_mtas), A.sum(1), -price_ab])
    beta2, se2, r2_2, resid2 = ols(X2, delta_ab)
    _print_regression("delta_MTA ~ const + |m| + (-price_MTA)",
                      n_mtas, ["const", "|m|", "alpha_1"], beta2, se2, r2_2, resid2)

    # (3) delta_MTA ~ const + alpha_1*(-price_MTA)
    X3 = np.column_stack([np.ones(n_mtas), -price_ab])
    beta3, se3, r2_3, resid3 = ols(X3, delta_ab)
    _print_regression("delta_MTA ~ const + (-price_MTA)",
                      n_mtas, ["const", "alpha_1"], beta3, se3, r2_3, resid3)

    # (4) delta_BTA ~ const + alpha_1*(-price_BTA)
    X4 = np.column_stack([np.ones(n_btas), -price_bta])
    beta4, se4, r2_4, resid4 = ols(X4, delta_bta)
    _print_regression("delta_BTA ~ const + (-price_BTA)",
                      n_btas, ["const", "alpha_1"], beta4, se4, r2_4, resid4)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    run_joint()
