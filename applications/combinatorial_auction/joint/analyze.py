#!/usr/bin/env python3
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import (
    load_bta_data, load_aggregation_matrix, load_ab_data,
)

JOINT_DIR = Path(__file__).parent
STANDALONE_DIR = APP_DIR / "standalone_blocks"


def extract_fe(result):
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    return theta[n_id : n_id + result["n_items"]]


def extract_joint_fe(result):
    theta = np.array(result["theta_hat"])
    n_id = result.get("n_id_mod", 1) + result.get("n_id_quad", 0)
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


def run_standalone(bta_result_file="result.json", ab_result_file="result.json"):
    bta_res = json.load(open(STANDALONE_DIR / bta_result_file))
    ab_res = json.load(open(STANDALONE_DIR / ab_result_file.replace("result", "ab_result")))
    fe_bta = extract_fe(bta_res)
    fe_ab = extract_fe(ab_res)

    raw = load_bta_data()
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_mtas = A.shape[0]
    mta_nums = ab_res["continental_mta_nums"]
    price_mta_c, price_ab, _ = load_prices(raw, mta_nums, A)

    Y = A @ fe_bta - fe_ab
    X = np.column_stack([np.ones(n_mtas), A.sum(1), price_mta_c - price_ab])
    beta, se, r2, resid = ols(X, Y)
    _print_regression("Cross-auction (standalone)", n_mtas, ["const", "|m|", "alpha"], beta, se, r2, resid)
    return dict(beta=beta, se=se, r2=r2, col_names=["const", "|m|", "alpha"], resid=resid)


def run_joint(result_file="result.json"):
    joint = json.load(open(JOINT_DIR / result_file))
    fe_bta, fe_ab = extract_joint_fe(joint)

    raw = load_bta_data()
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_mtas = A.shape[0]
    mta_nums = joint["continental_mta_nums"]
    price_mta_c, price_ab, _ = load_prices(raw, mta_nums, A)

    Y = A @ fe_bta - fe_ab
    X = np.column_stack([np.ones(n_mtas), A.sum(1), price_mta_c - price_ab])
    beta, se, r2, resid = ols(X, Y)
    _print_regression("Cross-auction (joint)", n_mtas, ["const", "|m|", "alpha"], beta, se, r2, resid)
    return dict(beta=beta, se=se, r2=r2, col_names=["const", "|m|", "alpha"], resid=resid)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("\n-- Joint --")
    run_joint()
