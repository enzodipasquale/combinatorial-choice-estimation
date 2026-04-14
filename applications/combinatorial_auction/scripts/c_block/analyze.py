#!/usr/bin/env python3
"""Point-estimate IV analysis: OLS + 2SLS on delta ~ alpha_0 - alpha_1 * price."""
import json, sys, argparse
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent
sys.path.insert(0, str(CBLOCK_DIR.parent.parent.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data
from applications.combinatorial_auction.data.iv import ols, tsls, build_distant_stats, load_iv_instruments

RESULTS_DIR = CBLOCK_DIR / "results"
POP_THRESHOLD = 500_000


def _print_regression(title, n, col_names, beta, se, r2, resid):
    print(f"\n{'='*60}")
    print(f"{title}  (N = {n})")
    print(f"{'='*60}")
    print(f"\n  {'Covariate':<20} {'Estimate':>12} {'SE':>12} {'t-stat':>10}")
    print(f"  {'-'*54}")
    for name, b, s in zip(col_names, beta, se):
        print(f"  {name:<20} {b:>12.6f} {s:>12.6f} {b/s:>10.3f}")
    print(f"\n  R2 = {r2:.4f}")
    print(f"  Residual std = {resid.std():.4f}")


def _run_iv_sample(label, mask, delta, price, zm, zs, zh):
    n = mask.sum()
    d_s, p_s = delta[mask], price[mask]

    X = np.column_stack([np.ones(n), -p_s])
    b, s, r2, res = ols(X, d_s)
    _print_regression(f"OLS {label}: delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b, s, r2, res)

    zm_s, zh_s = zm[mask], zh[mask]
    Z = np.column_stack([zm_s, zh_s])
    b_iv, s_iv, r2_iv, _ = tsls(X, d_s, Z)
    _print_regression(f"IV {label} (z=pop+hhinc, d>500): delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b_iv, s_iv, r2_iv,
                      delta[mask] - X @ b_iv)

    X_fs = np.column_stack([np.ones(n), zm_s, zh_s])
    _, _, r2_fs, _ = ols(X_fs, -p_s)
    f_stat = (r2_fs / 2) / ((1 - r2_fs) / (n - 3))
    print(f"  First-stage F = {f_stat:.1f},  R2 = {r2_fs:.4f}")
    print(f"  a0 in dollars = ${b_iv[0]/b_iv[1]*1e3:.1f}M per license")

    return b_iv


def run(result_file=None):
    result = json.load(open(result_file))
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_btas = result["n_btas"]
    delta = -(theta[n_id : n_id + n_btas])

    raw = load_bta_data()
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    zm, zs, zh = load_iv_instruments(raw, d=500)

    full = np.ones(n_btas, dtype=bool)
    rural = pop < POP_THRESHOLD

    print(f"\n{'#'*60}")
    print(f"# MAIN SPECIFICATION: Full sample (N = {n_btas})")
    print(f"{'#'*60}")
    b_iv_main = _run_iv_sample("full", full, delta, price, zm, zs, zh)

    print(f"\n{'#'*60}")
    print(f"# ROBUSTNESS: Rural (pop90 < {POP_THRESHOLD:,}, N = {rural.sum()})")
    print(f"{'#'*60}")
    _run_iv_sample("rural", rural, delta, price, zm, zs, zh)

    return b_iv_main


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", nargs="?",
                        default=str(RESULTS_DIR / "boot" / "result.json"))
    args = parser.parse_args()
    print(f">>> Analyzing: {args.result_file}")
    run(result_file=args.result_file)
