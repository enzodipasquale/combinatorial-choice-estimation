#!/usr/bin/env python3
import json, sys
import numpy as np
from pathlib import Path

SPECS_DIR = Path(__file__).parent.parent
APP_DIR = SPECS_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

CBLOCK_DIR = Path(__file__).parent
POP_THRESHOLD = 500_000


def ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def tsls(X, y, Z):
    """Two-stage least squares. X = [exog | endog], Z = [exog | instruments]."""
    Pz = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.lstsq(Pz, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(Pz.T @ X)
    se = np.sqrt(np.abs(np.diag(cov)))
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
    """For each BTA j, compute mean and std of var at BTAs farther than d."""
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


def _run_iv_block(label, mask, delta, price, dist_thresholds,
                  iv_pop_mean, iv_pop_std, iv_hhinc_mean):
    """Run OLS and IV regressions for a given subsample."""
    n = mask.sum()
    d_s, p_s = delta[mask], price[mask]

    # OLS: delta ~ const + (-price)
    X = np.column_stack([np.ones(n), -p_s])
    b, s, r2, res = ols(X, d_s)
    _print_regression(f"OLS {label}: delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b, s, r2, res)

    # IV at each distance threshold
    for d in dist_thresholds:
        zm = iv_pop_mean[d][mask]
        zs = iv_pop_std[d][mask]
        zh = iv_hhinc_mean[d][mask]

        X_endog = np.column_stack([np.ones(n), -p_s])

        # IV with avg_pop + std_pop (just-identified: 2 instruments, 2 endogenous)
        Z_iv = np.column_stack([zm, zs])  # no constant in Z: both regressors instrumented
        b_iv, s_iv, r2_iv, res_iv = tsls(X_endog, d_s, Z_iv)
        # first-stage F for -price on instruments
        X_fs = np.column_stack([np.ones(n), zm, zs])
        b_fs, _, r2_fs, _ = ols(X_fs, -p_s)
        f_stat = ((r2_fs) / 2) / ((1 - r2_fs) / (n - 3))
        _print_regression(f"IV {label} (z=avg_pop+std_pop, d>{d}km): delta ~ const + (-price)",
                          n, ["const", "alpha_1"], b_iv, s_iv, r2_iv, res_iv)
        print(f"  First-stage F = {f_stat:.1f},  R2 = {r2_fs:.4f}")

        # IV with avg_pop + avg_hhinc (just-identified, less collinear)
        Z_iv2 = np.column_stack([zm, zh])
        b_iv2, s_iv2, r2_iv2, res_iv2 = tsls(X_endog, d_s, Z_iv2)
        X_fs2 = np.column_stack([np.ones(n), zm, zh])
        _, _, r2_fs2, _ = ols(X_fs2, -p_s)
        f_stat2 = ((r2_fs2) / 2) / ((1 - r2_fs2) / (n - 3))
        _print_regression(f"IV {label} (z=avg_pop+avg_hhinc, d>{d}km): delta ~ const + (-price)",
                          n, ["const", "alpha_1"], b_iv2, s_iv2, r2_iv2, res_iv2)
        print(f"  First-stage F = {f_stat2:.1f},  R2 = {r2_fs2:.4f}")

        # IV with avg_pop + std_pop + avg_hhinc (overidentified)
        Z_iv3 = np.column_stack([zm, zs, zh])
        b_iv3, s_iv3, r2_iv3, res_iv3 = tsls(X_endog, d_s, Z_iv3)
        X_fs3 = np.column_stack([np.ones(n), zm, zs, zh])
        _, _, r2_fs3, _ = ols(X_fs3, -p_s)
        f_stat3 = ((r2_fs3) / 3) / ((1 - r2_fs3) / (n - 4))
        _print_regression(f"IV {label} (z=avg_pop+std_pop+avg_hhinc, d>{d}km): delta ~ const + (-price)",
                          n, ["const", "alpha_1"], b_iv3, s_iv3, r2_iv3, res_iv3)
        print(f"  First-stage F = {f_stat3:.1f},  R2 = {r2_fs3:.4f}")


def run_valuations(result_file="result_FE.json", alpha_0=-2.495269, alpha_1=40.678871):
    """
    Net surplus ($B) = u_hat / alpha_1
    """
    result = json.load(open(CBLOCK_DIR / result_file))
    u_hat = np.array(result["u_hat"])
    n_obs = result["n_obs"]
    n_agents = len(u_hat)
    n_sim = n_agents // n_obs

    net_surplus = u_hat / alpha_1
    net_m = net_surplus.reshape(n_obs, n_sim).mean(1)

    print(f"\n{'='*60}")
    print(f"VALUATIONS  (a0={alpha_0:.4f}, a1={alpha_1:.4f}, "
          f"n_sim={n_sim}, n_obs={n_obs})")
    print(f"{'='*60}")
    obj = result["objective"]
    eps_net = obj / (n_sim * alpha_1)
    total_net = net_m.sum()
    obs_net = total_net - eps_net

    print(f"  total net surplus  = ${total_net:>10.4f}B")
    print(f"    observable part  = ${obs_net:>10.4f}B")
    print(f"    epsilon part     = ${eps_net:>10.4f}B  (= obj / n_sim / a1)")
    print(f"  mean net surplus   = ${net_m.mean():>10.4f}B")
    print(f"  median net surplus = ${np.median(net_m):>10.4f}B")

    # ── observable part decomposition using observed bundles ──
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

    obs_revenue = obs_price_paid.sum()                        # (1) -obs_revenue
    xi_part = (b_obs @ xi).sum() / alpha_1                    # (2) xi component
    a0_part = alpha_0 / alpha_1 * b_obs.sum()                 # (3) alpha_0 component
    gamma_part = obs_net - (-obs_revenue) - xi_part - a0_part # (4) residual

    print(f"\n  Observable part decomposition (at observed bundles):")
    print(f"    (1) -observed revenue  = ${-obs_revenue:>10.4f}B")
    print(f"    (2) xi component       = ${xi_part:>10.4f}B")
    print(f"    (3) alpha_0 component  = ${a0_part:>10.4f}B")
    print(f"    (4) gamma residual     = ${gamma_part:>10.4f}B")
    print(f"    sum (1)+(2)+(3)+(4)    = ${(-obs_revenue + xi_part + a0_part + gamma_part):>10.4f}B")

    top5 = np.argsort(net_m)[::-1][:5]
    print(f"\n  Top 5 by net surplus:")
    print(f"  {'#':<3} {'Bidder':<40} {'Net ($B)':>10} {'Price ($B)':>10} {'|b|':>4}")
    print(f"  {'-'*70}")
    for r, i in enumerate(top5, 1):
        print(f"  {r:<3} {bidder_names[i]:<40} {net_m[i]:>10.4f} {obs_price_paid[i]:>10.4f} {obs_bundle_size[i]:>4d}")


def run(result_file="result_FE.json"):
    result = json.load(open(CBLOCK_DIR / result_file))
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

    # ── Full sample OLS ─────────────────────────────────────────────
    X_full = np.column_stack([np.ones(n_btas), -price])
    b, s, r2, res = ols(X_full, delta)
    _print_regression("OLS full: delta ~ const + (-price)",
                      n_btas, ["const", "alpha_1"], b, s, r2, res)

    # ── Rural only ──────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# RURAL (pop90 < {POP_THRESHOLD:,}): {rural.sum()} BTAs")
    print(f"{'#'*60}")
    _run_iv_block("rural", rural, delta, price, dist_thresholds,
                  iv_pop_mean, iv_pop_std, iv_hhinc_mean)



if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    run()
    run_valuations()
