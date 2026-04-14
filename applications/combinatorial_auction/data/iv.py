"""IV regression utilities for second-stage estimation."""
import numpy as np


def robust_cov(X, resid):
    """HC1 sandwich covariance."""
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
    cov = robust_cov(X, resid)
    se = np.sqrt(np.abs(np.diag(cov)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def tsls(X, y, Z):
    Pz = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.lstsq(Pz, y, rcond=None)[0]
    resid = y - X @ beta
    cov = robust_cov(Pz, resid)
    se = np.sqrt(np.abs(np.diag(cov)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid


def build_distant_stats(var, geo, dist_thresholds):
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


def load_iv_instruments(raw, d=500):
    """Build pop and hhinc instruments at distance threshold d."""
    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    hhinc = raw["bta_data"]["hhinc35k"].to_numpy().astype(float)
    geo = raw["geo_distance"]
    iv_pop_mean, iv_pop_std = build_distant_stats(pop, geo, [d])
    iv_hhinc_mean, _ = build_distant_stats(hhinc, geo, [d])
    return iv_pop_mean[d], iv_pop_std[d], iv_hhinc_mean[d]


def run_2sls(delta, price, zm, zh):
    """2SLS: delta ~ alpha_0 - alpha_1 * price. Returns (a0, a1, se_a0, se_a1, r2)."""
    n = len(delta)
    X = np.column_stack([np.ones(n), -price])
    Z = np.column_stack([zm, zh])
    b, se, r2, _ = tsls(X, delta, Z)
    return b[0], b[1], se[0], se[1], r2
