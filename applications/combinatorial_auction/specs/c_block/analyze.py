#!/usr/bin/env python3
import json, sys
import numpy as np
from pathlib import Path

SPECS_DIR = Path(__file__).parent.parent
APP_DIR = SPECS_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC
from applications.combinatorial_auction.data.prepare import _build_features

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
    """Compute bidder valuations in $B using IV price coefficient."""
    result = json.load(open(CBLOCK_DIR / result_file))
    theta = np.array(result["theta_hat"])
    n_id = result["n_id_mod"]
    n_btas = result["n_btas"]
    n_obs = result["n_obs"]

    beta = theta[:n_id]                           # elig_pop coefficient
    theta_fe = theta[n_id : n_id + n_btas]        # item FEs
    gamma = theta[n_id + n_btas:]                  # quadratic coefficients
    delta = -theta_fe                              # delta_j = -theta_j^FE

    raw = load_bta_data()
    ctx = build_context(raw)
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    obs = ctx["c_obs_bundles"].astype(float)       # (n_obs, n_btas)

    # recover xi
    xi = delta - alpha_0 + alpha_1 * price         # xi_j = delta_j - alpha_0 + alpha_1*p_j

    # structural utility at observed bundle (no epsilon)
    # V_i(obs_i) = id_mod_contribution + item_fe_contribution + quad_contribution

    # 1. id_modular: sum_j obs_ij * elig_i * pop_j * beta
    id_mod = _build_features(MODULAR, ["elig_pop"], ctx)  # (n_obs, n_btas, 1)
    id_mod_at_obs = np.einsum('ijk,ij->ik', id_mod, obs)  # (n_obs, 1)
    id_contribution = (id_mod_at_obs @ beta)               # (n_obs,)

    # 2. item FE: sum_j obs_ij * delta_j
    fe_contribution = obs @ delta                           # (n_obs,)

    # 3. quadratic: sum_k gamma_k * obs_i' Q_k obs_i
    quad_names = ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]
    Q = _build_features(QUADRATIC, quad_names, ctx)        # (n_btas, n_btas, 4)
    quad_contribution = np.einsum('ij,jlk,il,k->i', obs, Q, obs, gamma)  # (n_obs,)

    # total structural utility
    V_obs = id_contribution + fe_contribution + quad_contribution

    # decompose fe_contribution: delta_j = alpha_0 - alpha_1*p_j + xi_j
    n_items_won = obs.sum(1)                                # |b_i|
    price_paid = obs @ price                                # sum_j obs_ij * p_j ($B)
    alpha0_contribution = n_items_won * alpha_0
    price_contribution = -alpha_1 * price_paid              # -alpha_1 * p in utility
    xi_contribution = obs @ xi                              # sum_j obs_ij * xi_j

    # gross value = everything except price term, in $B
    gross_structural = (id_contribution + alpha0_contribution + xi_contribution
                        + quad_contribution) / alpha_1
    net_structural = V_obs / alpha_1                        # = gross - price_paid

    # u_hat: includes epsilon, at optimal bundle (may differ from observed)
    u_hat = np.array(result["u_hat"])
    n_sim = len(u_hat) // n_obs
    u_hat_mean = u_hat.reshape(n_obs, n_sim).mean(1)
    net_surplus = u_hat_mean / alpha_1

    # only winners (bidders who won at least one item)
    winners = n_items_won > 0

    print(f"\n{'='*60}")
    print(f"BIDDER VALUATIONS (alpha_0={alpha_0:.4f}, alpha_1={alpha_1:.4f})")
    print(f"{'='*60}")
    print(f"\n  {n_sim} simulation(s), {int(winners.sum())} winners / {n_obs} bidders")

    print(f"\n  --- All bidders (N={n_obs}) ---")
    print(f"  {'Metric':<35} {'Mean':>10} {'Median':>10} {'Total':>12}")
    print(f"  {'-'*67}")
    print(f"  {'Net surplus (u_hat/a1, $B)':<35} {net_surplus.mean():>10.6f} {np.median(net_surplus):>10.6f} {net_surplus.sum():>12.4f}")
    print(f"  {'Net structural (V/a1, $B)':<35} {net_structural.mean():>10.6f} {np.median(net_structural):>10.6f} {net_structural.sum():>12.4f}")
    print(f"  {'Gross structural ($B)':<35} {gross_structural.mean():>10.6f} {np.median(gross_structural):>10.6f} {gross_structural.sum():>12.4f}")
    print(f"  {'Price paid ($B)':<35} {price_paid.mean():>10.6f} {np.median(price_paid):>10.6f} {price_paid.sum():>12.4f}")

    print(f"\n  --- Winners only (N={int(winners.sum())}) ---")
    print(f"  {'Metric':<35} {'Mean':>10} {'Median':>10} {'Total':>12}")
    print(f"  {'-'*67}")
    w = winners
    print(f"  {'Items won':<35} {n_items_won[w].mean():>10.1f} {np.median(n_items_won[w]):>10.1f} {n_items_won[w].sum():>12.0f}")
    print(f"  {'Net surplus (u_hat/a1, $B)':<35} {net_surplus[w].mean():>10.6f} {np.median(net_surplus[w]):>10.6f} {net_surplus[w].sum():>12.4f}")
    print(f"  {'Net structural (V/a1, $B)':<35} {net_structural[w].mean():>10.6f} {np.median(net_structural[w]):>10.6f} {net_structural[w].sum():>12.4f}")
    print(f"  {'Gross structural ($B)':<35} {gross_structural[w].mean():>10.6f} {np.median(gross_structural[w]):>10.6f} {gross_structural[w].sum():>12.4f}")
    print(f"  {'Price paid ($B)':<35} {price_paid[w].mean():>10.6f} {np.median(price_paid[w]):>10.6f} {price_paid[w].sum():>12.4f}")
    print(f"  {'Markup (gross/price - 1)':<35} {(gross_structural[w]/price_paid[w]).mean()-1:>10.4f} {np.median(gross_structural[w]/price_paid[w])-1:>10.4f}")

    print(f"\n  --- Aggregate ---")
    print(f"  Total auction revenue:  ${price_paid.sum():.4f}B")
    print(f"  Total gross value:      ${gross_structural.sum():.4f}B")
    print(f"  Total net surplus:      ${net_structural.sum():.4f}B")
    print(f"  Total surplus (w/ eps): ${net_surplus.sum():.4f}B")


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
