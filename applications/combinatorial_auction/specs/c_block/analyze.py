#!/usr/bin/env python3
import json, sys
import numpy as np
from pathlib import Path

SPECS_DIR = Path(__file__).parent.parent
APP_DIR = SPECS_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data

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
    Decompose bidder values using u_hat, b*, and the IV price coefficient.

    u_hat_si  = features(b*_si) @ theta_hat + eps_si(b*_si)

    The FE part of theta encodes  theta^FE_j = -delta_j  with
    delta_j = alpha_0 - alpha_1*p_j + xi_j,  so the price cost in
    utility space is  alpha_1 * price(b*).

    Gross surplus ($B) = u_hat / alpha_1 + price(b*)
    Net   surplus ($B) = u_hat / alpha_1
    Price paid    ($B) = b* @ price
    """
    result = json.load(open(CBLOCK_DIR / result_file))
    u_hat = np.array(result["u_hat"])                        # (n_agents,)
    bundles = np.array(result["predicted_bundles"])             # (n_agents, n_btas)
    n_obs = result["n_obs"]
    n_agents = len(u_hat)
    n_sim = n_agents // n_obs

    theta = np.array(result["theta_hat"])

    raw = load_bta_data()
    price = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9  # (n_btas,) $B

    # ── structural / epsilon from cached oracle outputs ────
    covariates = np.array(result["predicted_covariates"])     # (n_agents, n_covariates)
    errors = np.array(result["predicted_errors"])             # (n_agents,)
    structural = covariates @ theta                           # (n_agents,)
    epsilon = errors                                          # (n_agents,)

    # ── per agent-simulation quantities ──────────────────────
    price_paid = bundles @ price                             # (n_agents,) $B
    n_items = bundles.sum(1)                                 # (n_agents,)
    net_surplus = u_hat / alpha_1                            # (n_agents,) $B
    gross_surplus = net_surplus + price_paid                  # (n_agents,) $B

    # ── average over simulations → per-bidder ────────────────
    price_paid_m = price_paid.reshape(n_obs, n_sim).mean(1)
    n_items_m = n_items.reshape(n_obs, n_sim).mean(1)
    net_m = net_surplus.reshape(n_obs, n_sim).mean(1)
    gross_m = gross_surplus.reshape(n_obs, n_sim).mean(1)

    winners = n_items_m > 0

    print(f"\n{'='*60}")
    print(f"BIDDER VALUATIONS  (a0={alpha_0:.4f}, a1={alpha_1:.4f})")
    print(f"{'='*60}")
    print(f"  {n_sim} simulation(s),  {int(winners.sum())} winners / {n_obs} bidders")

    def _block(label, mask):
        n = mask.sum()
        print(f"\n  --- {label} (N={n}) ---")
        print(f"  {'Metric':<35} {'Mean':>12} {'Median':>12} {'Total':>14}")
        print(f"  {'-'*73}")
        for name, arr in [("Items in b*",     n_items_m),
                          ("Gross value ($B)", gross_m),
                          ("Price paid ($B)",  price_paid_m),
                          ("Net surplus ($B)", net_m)]:
            v = arr[mask]
            print(f"  {name:<35} {v.mean():>12.6f} {np.median(v):>12.6f} {v.sum():>14.4f}")
        # markup only for bidders with positive price
        pp = price_paid_m[mask]
        gv = gross_m[mask]
        pos = pp > 0
        if pos.any():
            markup = gv[pos] / pp[pos] - 1
            print(f"  {'Markup (gross/price - 1)':<35} {markup.mean():>12.4f} {np.median(markup):>12.4f}")

    _block("All bidders", np.ones(n_obs, dtype=bool))
    _block("Winners", winners)

    # ── 4-component decomposition of u_hat ─────────────────────
    #
    # u_hat_si = max_b { beta_contrib(b) + gamma_contrib(b)
    #                   + alpha_0*|b| + xi*b
    #                   - alpha_1*price(b)
    #                   + eps_si(b) }
    #
    # At b* this gives exactly:
    # u_hat = [beta_gamma_contrib] + [alpha_0*|b*| + xi*b*] + [-alpha_1*price(b*)] + epsilon
    #
    # (1) beta + gamma contribution (id_modular + quadratic)
    n_id = result["n_id_mod"]
    n_btas_r = result["n_btas"]
    beta_gamma_contrib = (covariates[:, :n_id] @ theta[:n_id]
                          + covariates[:, n_id + n_btas_r:] @ theta[n_id + n_btas_r:])

    # (2) non-price FE: alpha_0*|b*| + xi*b*
    theta_fe = theta[n_id : n_id + n_btas_r]
    delta = -theta_fe                                            # delta_j = -theta^FE_j
    xi = delta - alpha_0 + alpha_1 * price                      # xi_j = delta_j - alpha_0 + alpha_1*p_j
    non_price_fe = alpha_0 * n_items + bundles @ xi              # alpha_0*|b*| + xi*b*

    # (3) price FE: -alpha_1 * price(b*)
    price_fe = -alpha_1 * price_paid

    # (4) epsilon (private shock)
    # already have: epsilon = errors

    # sanity check: (1)+(2)+(3)+(4) should equal u_hat
    recon = beta_gamma_contrib + non_price_fe + price_fe + epsilon
    print(f"\n  Reconstruction error: {np.abs(recon - u_hat).max():.2e}")

    # convert to gross $B: divide by alpha_1 and add back price for gross
    # gross = u_hat/alpha_1 + price = [(1)+(2)+(3)+(4)]/alpha_1 + price
    #       = (1)/a1 + (2)/a1 + [(3)/a1 + price] + (4)/a1
    #       = (1)/a1 + (2)/a1 + 0              + (4)/a1
    # note: (3)/a1 + price = -price + price = 0, so price cancels!
    gross_beta_gamma = beta_gamma_contrib / alpha_1
    gross_non_price_fe = non_price_fe / alpha_1     # (alpha_0*|b*| + xi*b*) / alpha_1
    gross_epsilon = epsilon / alpha_1

    # average over simulations
    g_bg_m = gross_beta_gamma.reshape(n_obs, n_sim).mean(1)
    g_npfe_m = gross_non_price_fe.reshape(n_obs, n_sim).mean(1)
    g_eps_m = gross_epsilon.reshape(n_obs, n_sim).mean(1)

    obs_revenue = price.sum()
    print(f"\n  --- Aggregate ---")
    print(f"  Observed revenue:    ${obs_revenue:.4f}B")
    print(f"  Predicted revenue:   ${price_paid_m.sum():.4f}B")
    print(f"  Total gross value:   ${gross_m.sum():.4f}B")
    print(f"  Total net surplus:   ${net_m.sum():.4f}B")
    print(f"\n  --- Gross value decomposition ---")
    print(f"  (1) beta+gamma:      ${g_bg_m.sum():.4f}B  ({g_bg_m.sum()/gross_m.sum():.1%})")
    print(f"  (2) alpha_0+xi:      ${g_npfe_m.sum():.4f}B  ({g_npfe_m.sum()/gross_m.sum():.1%})")
    print(f"  (3) price FE:        cancels with price_paid (= $0)")
    print(f"  (4) epsilon:         ${g_eps_m.sum():.4f}B  ({g_eps_m.sum()/gross_m.sum():.1%})")
    print(f"  Sum (1)+(2)+(4):     ${(g_bg_m.sum()+g_npfe_m.sum()+g_eps_m.sum()):.4f}B")

    # ── same decomposition without xi (prediction case) ──────
    # xi_j are 2SLS residuals, not known ex ante.
    # Set xi=0: non-price FE becomes just alpha_0*|b*|,
    # and xi*b* gets absorbed into a residual together with epsilon.
    gross_alpha0_only = (alpha_0 * n_items) / alpha_1              # $B
    gross_xi_plus_eps = (bundles @ xi + epsilon) / alpha_1         # $B  (residual)
    g_a0_m = gross_alpha0_only.reshape(n_obs, n_sim).mean(1)
    g_xe_m = gross_xi_plus_eps.reshape(n_obs, n_sim).mean(1)

    print(f"\n  --- Gross value decomposition (without xi, prediction case) ---")
    print(f"  (1) beta+gamma:      ${g_bg_m.sum():.4f}B  ({g_bg_m.sum()/gross_m.sum():.1%})")
    print(f"  (2) alpha_0 only:    ${g_a0_m.sum():.4f}B  ({g_a0_m.sum()/gross_m.sum():.1%})")
    print(f"  (3) price FE:        cancels with price_paid (= $0)")
    print(f"  (4) xi+epsilon:      ${g_xe_m.sum():.4f}B  ({g_xe_m.sum()/gross_m.sum():.1%})")
    print(f"  Sum (1)+(2)+(4):     ${(g_bg_m.sum()+g_a0_m.sum()+g_xe_m.sum()):.4f}B")

    # ── top 5 bidders by gross value ─────────────────────────
    bidder_names = raw["bidder_data"]["co_name"].values
    top5 = np.argsort(gross_m)[::-1][:5]
    print(f"\n  --- Top 5 bidders by gross value ---")
    print(f"  {'#':<4} {'Bidder':<30} {'Gross ($B)':>12} {'Price ($B)':>12} {'Net ($B)':>12} {'Items':>6}")
    print(f"  {'-'*76}")
    for rank, i in enumerate(top5, 1):
        print(f"  {rank:<4} {bidder_names[i]:<30} {gross_m[i]:>12.6f} {price_paid_m[i]:>12.6f} {net_m[i]:>12.6f} {n_items_m[i]:>6.1f}")


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
