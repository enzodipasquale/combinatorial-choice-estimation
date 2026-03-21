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


def _run_iv(label, n, d_s, p_s, zm, zs, zh, instruments, d):
    """Run a single IV regression. Returns (spec_label, beta, se, r2, f_stat, r2_fs)."""
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
        X_fs = Z  # already has constant
    else:
        X_fs = np.column_stack([np.ones(n)] + [Z[:, k] for k in range(n_inst)])
    _, _, r2_fs, _ = ols(X_fs, -p_s)
    f_stat = (r2_fs / n_inst) / ((1 - r2_fs) / (n - n_inst - 1))
    spec_label = f"IV z={z_label} d>{d}"
    return spec_label, b_iv, s_iv, r2_iv, f_stat, r2_fs


# ── IV spec registry ──────────────────────────────────────────────
# (instruments, dist_threshold_km)
PREFERRED_IV = ("pop_hhinc", 500)
OTHER_IV_SPECS = [
    ("pop",        500),
    ("pop_std",    500),
    ("all",        500),
    ("pop",       1000),
    ("pop_hhinc", 1000),
    ("pop_std",   1000),
    ("all",       1000),
    ("pop",       1500),
    ("pop_hhinc", 1500),
    ("pop_std",   1500),
    ("all",       1500),
    ("pop",       2000),
    ("pop_hhinc", 2000),
    ("pop_std",   2000),
    ("all",       2000),
]


def _run_iv_block(label, mask, delta, price, dist_thresholds,
                  iv_pop_mean, iv_pop_std, iv_hhinc_mean):
    """Run OLS and preferred IV in detail, then summary table of all IV specs."""
    n = mask.sum()
    d_s, p_s = delta[mask], price[mask]

    # ── OLS ───────────────────────────────────────────────────────
    X = np.column_stack([np.ones(n), -p_s])
    b, s, r2, res = ols(X, d_s)
    _print_regression(f"OLS {label}: delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b, s, r2, res)

    # ── Preferred IV (full output) ────────────────────────────────
    def _get_zs(d):
        return iv_pop_mean[d][mask], iv_pop_std[d][mask], iv_hhinc_mean[d][mask]

    inst, d = PREFERRED_IV
    zm, zs, zh = _get_zs(d)
    spec, b_iv, s_iv, r2_iv, f_stat, r2_fs = _run_iv(label, n, d_s, p_s, zm, zs, zh, inst, d)
    _print_regression(f"IV {label} ({spec}): delta ~ const + (-price)",
                      n, ["const", "alpha_1"], b_iv, s_iv, r2_iv, delta[mask] - np.column_stack([np.ones(n), -p_s]) @ b_iv)
    print(f"  First-stage F = {f_stat:.1f},  R2 = {r2_fs:.4f}")

    # ── Summary table of all IV specs ─────────────────────────────
    rows = [(spec, b_iv, s_iv, r2_iv, f_stat)]
    for inst, d in OTHER_IV_SPECS:
        zm, zs, zh = _get_zs(d)
        sp, bi, si, ri, fi, _ = _run_iv(label, n, d_s, p_s, zm, zs, zh, inst, d)
        rows.append((sp, bi, si, ri, fi))

    S = 24  # spec column width
    print(f"\n{'='*90}")
    print(f"All IV specs {label}  (N = {n})")
    print(f"{'='*90}")
    print(f"  {'Spec':<{S}} {'a0':>8} {'se':>8} {'t':>7}  {'a1':>8} {'se':>8} {'t':>7}  {'F':>6} {'R2':>6}")
    print(f"  {'-'*(S+8+8+7+2+8+8+7+2+6+6+4)}")
    for sp, bi, si, ri, fi in rows:
        t0, t1 = bi[0]/si[0], bi[1]/si[1]
        print(f"  {sp:<{S}} {bi[0]:>8.3f} {si[0]:>8.3f} {t0:>7.2f}  {bi[1]:>8.3f} {si[1]:>8.3f} {t1:>7.2f}  {fi:>6.1f} {ri:>6.3f}")

    return b_iv  # preferred IV coefficients: [alpha_0, alpha_1]


def run_valuations(result_file="result_FE.json", alpha_0=None, alpha_1=None):
    """
    Net surplus ($B) = u_hat / alpha_1
    """
    if alpha_0 is None or alpha_1 is None:
        raise ValueError("alpha_0 and alpha_1 must be provided (from IV regression)")
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
    print(f"    epsilon part     = ${eps_net:>10.4f}B")

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
    b_iv = _run_iv_block("rural", rural, delta, price, dist_thresholds,
                         iv_pop_mean, iv_pop_std, iv_hhinc_mean)
    return b_iv  # [alpha_0, alpha_1] from preferred IV



if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    b_iv = run()
    run_valuations(alpha_0=b_iv[0], alpha_1=b_iv[1])
