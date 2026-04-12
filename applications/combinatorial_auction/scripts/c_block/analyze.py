#!/usr/bin/env python3
import json, sys
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent
sys.path.insert(0, str(CBLOCK_DIR.parent.parent.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

RESULTS_DIR = CBLOCK_DIR / "results"
POP_THRESHOLD = 500_000


def _robust_cov(X, resid):
    n, k = X.shape
    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXinv = np.linalg.pinv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    return (n / (n - k)) * XtXinv @ meat @ XtXinv


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    cov = _robust_cov(X, resid)
    se = np.sqrt(np.abs(np.diag(cov)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def tsls(X, y, Z):
    Pz = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.lstsq(Pz, y, rcond=None)[0]
    resid = y - X @ beta
    cov = _robust_cov(Pz, resid)
    se = np.sqrt(np.abs(np.diag(cov)))
    n, k = X.shape
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


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


def _build_distant_stats(var, geo, dist_thresholds):
    means, stds = {}, {}
    for d in dist_thresholds:
        M = np.zeros(len(var))
        S = np.zeros(len(var))
        for j in range(len(var)):
            mask = geo[j] > d
            if mask.any():
                M[j] = var[mask].mean()
                S[j] = var[mask].std()
        means[d] = M
        stds[d] = S
    return means, stds


def _run_iv(label, n, d_s, p_s, zm, zs, zh, instruments, d):
    X = np.column_stack([np.ones(n), -p_s])

    if instruments == "pop":
        Z = np.column_stack([np.ones(len(zm)), zm])
        z_label, n_inst = "pop", 1
    elif instruments == "pop_hhinc":
        Z = np.column_stack([zm, zh])
        z_label, n_inst = "pop+hhinc", 2
    elif instruments == "pop_std":
        Z = np.column_stack([zm, zs])
        z_label, n_inst = "pop+std", 2
    elif instruments == "all":
        Z = np.column_stack([zm, zs, zh])
        z_label, n_inst = "pop+std+hhinc", 3

    b_iv, s_iv, r2_iv, res_iv = tsls(X, d_s, Z)
    if instruments == "pop":
        X_fs = Z  # [ones, zm] — already has constant, 1 excluded instrument
        n_excl = 1
    else:
        X_fs = np.column_stack([np.ones(n)] + [Z[:, k] for k in range(n_inst)])
        n_excl = n_inst
    _, _, r2_fs, _ = ols(X_fs, -p_s)
    f_stat = (r2_fs / n_excl) / ((1 - r2_fs) / (n - n_excl - 1))
    spec_label = f"IV z={z_label} d>{d}"
    return spec_label, b_iv, s_iv, r2_iv, f_stat, r2_fs


PREFERRED_IV = ("pop_hhinc", 500)


def _run_iv_sample(label, mask, delta, price, iv_pop_mean, iv_pop_std, iv_hhinc_mean):
    n = mask.sum()
    d_s, p_s = delta[mask], price[mask]

    X = np.column_stack([np.ones(n), -p_s])
    b, s, r2, res = ols(X, d_s)
    _print_regression(f"OLS {label}: delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b, s, r2, res)

    zm, zh = iv_pop_mean[500][mask], iv_hhinc_mean[500][mask]
    spec, b_iv, s_iv, r2_iv, f_stat, r2_fs = _run_iv(
        label, n, d_s, p_s, zm, iv_pop_std[500][mask], zh, "pop_hhinc", 500)
    _print_regression(f"IV {label} (z=pop+hhinc, d>500): delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b_iv, s_iv, r2_iv,
                      delta[mask] - X @ b_iv)
    print(f"  First-stage F = {f_stat:.1f},  R2 = {r2_fs:.4f}")
    print(f"  a0 in dollars = ${b_iv[0]/b_iv[1]*1e3:.1f}M per license")

    return b_iv


def run_valuations(result_file=None, alpha_0=None, alpha_1=None):
    if alpha_0 is None or alpha_1 is None:
        raise ValueError("alpha_0 and alpha_1 must be provided (from IV regression)")
    result = json.load(open(result_file))
    u_hat = np.array(result["u_hat"])
    n_obs = result["n_obs"]
    n_agents = len(u_hat)
    n_sim = n_agents // n_obs

    net_surplus = u_hat / alpha_1
    net_m = net_surplus.reshape(n_obs, n_sim).mean(1)

    raw = load_bta_data()
    ctx = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    b_obs = ctx["c_obs_bundles"]                              # (n_obs, n_btas)
    obs_bundle_size = b_obs.sum(1)
    obs_price_paid = b_obs @ price
    bidder_names = raw["bidder_data"]["co_name"].values

    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_btas = result["n_btas"]
    theta_fe = theta[n_id : n_id + n_btas]
    delta = -theta_fe
    xi = delta - alpha_0 + alpha_1 * price

    obs_revenue = obs_price_paid.sum()
    xi_part = (b_obs @ xi).sum() / alpha_1
    a0_part = alpha_0 / alpha_1 * b_obs.sum()
    obs_net = -obs_revenue + xi_part + a0_part

    print(f"\n{'='*60}")
    print(f"VALUATIONS  (a0={alpha_0:.4f}, a1={alpha_1:.4f}, "
          f"n_sim={n_sim}, n_obs={n_obs})")
    print(f"{'='*60}")
    total_net = net_m.sum()

    print(f"  total net surplus    = ${total_net:>10.4f}B")
    print(f"  observed revenue     = ${obs_revenue:>10.4f}B")
    print(f"\n  Observable part decomposition (at observed bundles):")
    print(f"    (1) -observed revenue  = ${-obs_revenue:>10.4f}B")
    print(f"    (2) xi component       = ${xi_part:>10.4f}B")
    print(f"    (3) alpha_0 component  = ${a0_part:>10.4f}B")

    elig = raw["bidder_data"]["pops_eligible"].to_numpy().astype(float)
    assets = raw["bidder_data"]["assets"].to_numpy().astype(float)
    revenues = raw["bidder_data"]["revenues"].to_numpy().astype(float)
    W = 30  # bidder name column width

    def _top5(title, idx):
        print(f"\n  {title}:")
        print(f"  {'#':<3} {'Bidder':<{W}} {'Elig (M)':>8} {'Net ($M)':>9} {'Price ($M)':>10} {'|b|':>4}")
        print(f"  {'-'*(3+W+8+9+10+4+5)}")
        for rank, i in enumerate(idx, 1):
            name = bidder_names[i][:W]
            print(f"  {rank:<3} {name:<{W}} {elig[i]/1e6:>8.1f} {net_m[i]*1e3:>9.1f} {obs_price_paid[i]*1e3:>10.1f} {obs_bundle_size[i]:>4d}")

    _top5("Top 5 by net surplus", np.argsort(net_m)[::-1][:5])
    _top5("Top 5 by eligibility", np.argsort(elig)[::-1][:5])
    _top5("Top 5 by assets", np.argsort(assets)[::-1][:5])
    _top5("Top 5 by revenues", np.argsort(revenues)[::-1][:5])


def run(result_file=None):
    result = json.load(open(result_file))
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_btas = result["n_btas"]
    delta = -(theta[n_id : n_id + n_btas])

    raw = load_bta_data()
    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    hhinc = raw["bta_data"]["hhinc35k"].to_numpy().astype(float)
    geo = raw["geo_distance"]
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9

    rural = pop < POP_THRESHOLD
    dist_thresholds = [500, 1000, 1500, 2000]
    iv_pop_mean, iv_pop_std = _build_distant_stats(pop, geo, dist_thresholds)
    iv_hhinc_mean, _ = _build_distant_stats(hhinc, geo, dist_thresholds)

    full = np.ones(n_btas, dtype=bool)
    print(f"\n{'#'*60}")
    print(f"# MAIN SPECIFICATION: Full sample (N = {n_btas})")
    print(f"{'#'*60}")
    b_iv_main = _run_iv_sample("full", full, delta, price,
                                iv_pop_mean, iv_pop_std, iv_hhinc_mean)

    print(f"\n{'#'*60}")
    print(f"# ROBUSTNESS: Rural (pop90 < {POP_THRESHOLD:,}, N = {rural.sum()})")
    print(f"{'#'*60}")
    b_iv_rural = _run_iv_sample("rural", rural, delta, price,
                                 iv_pop_mean, iv_pop_std, iv_hhinc_mean)

    return b_iv_main  # main spec coefficients



if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", nargs="?",
                        default=str(RESULTS_DIR / "boot" / "result.json"),
                        help="Path to result JSON file")
    args = parser.parse_args()
    print(f">>> Analyzing: {args.result_file}")
    b_iv = run(result_file=args.result_file)
    run_valuations(result_file=args.result_file, alpha_0=b_iv[0], alpha_1=b_iv[1])
