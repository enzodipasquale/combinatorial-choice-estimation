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


def load_blp_instruments(raw, pop_threshold=500_000):
    """Build BLP-style instruments: characteristics of other BTAs in same MTA.

    Returns (blp_arrays dict, rural mask, percapin array) for the continental BTAs
    in the same order as raw['bta_data'].
    """
    import pandas as pd
    from pathlib import Path

    DATA = Path(__file__).parent / "datasets" / "114402-V1" / "Replication-Fox-and-Bajari" / "data"
    census = pd.read_csv(DATA / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    bta_mta = census[["BTA", "MTA"]].dropna().copy()
    bta_mta.columns = ["bta", "mta"]
    bta_mta["bta"] = bta_mta["bta"].astype(int)
    bta_mta["mta"] = bta_mta["mta"].astype(int)

    bta_data = pd.read_csv(DATA / "btadata_2004_03_12_1.csv")
    bta_stats = pd.read_csv(DATA / "btastatsexport.csv")
    df = bta_data[["bta", "pop90", "percapin", "hhinc35k", "density", "imwl"]].merge(
        bta_stats[["bta", "grow9099"]], on="bta"
    ).merge(bta_mta, on="bta")

    bta_order = raw["bta_data"]["bta"].values
    cont_set = set(bta_order)
    df = df[df["bta"].isin(cont_set)].reset_index(drop=True)

    blp_vars = ["pop90", "percapin", "density", "hhinc35k", "grow9099", "imwl"]
    for var in blp_vars:
        s = df.groupby("mta")[var].transform("sum")
        c = df.groupby("mta")[var].transform("count")
        df[f"blp_{var}"] = (s - df[var]) / (c - 1)

    blp_cols = [f"blp_{v}" for v in blp_vars]
    arrays = {}
    for col in blp_cols + ["percapin"]:
        d = dict(zip(df["bta"], df[col]))
        arrays[col] = np.array([d.get(b, np.nan) for b in bta_order])

    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    rural = pop < pop_threshold

    return arrays, rural


def run_2sls_blp(delta, price, raw, pop_threshold=500_000):
    """2SLS with BLP instruments, pop control, rural sample.

    delta ~ const + pop + (-price)
    Endogenous: pop, price.
    Excluded IVs: percapin + BLP characteristics of other BTAs in same MTA.

    Returns dict with a0, a1, b_pop, se's, r2, n, demand_controls.
    """
    arrays, rural = load_blp_instruments(raw, pop_threshold)

    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    percapin = arrays["percapin"]
    blp_cols = [f"blp_{v}" for v in ["pop90", "percapin", "density", "hhinc35k", "grow9099", "imwl"]]

    valid = rural.copy()
    for c in blp_cols:
        valid &= ~np.isnan(arrays[c])
    valid &= ~np.isnan(percapin)
    n = valid.sum()

    d, p = delta[valid], price[valid]
    pop_v, inc_v = pop[valid], percapin[valid]

    X = np.column_stack([np.ones(n), pop_v, -p])
    z_excl = np.column_stack([inc_v] + [arrays[c][valid] for c in blp_cols])
    Z = np.column_stack([np.ones(n), z_excl])

    b, se, r2, resid = tsls(X, d, Z)

    return {
        "a0": b[0], "a1": b[2], "b_pop": b[1],
        "se_a0": se[0], "se_a1": se[2], "se_pop": se[1],
        "r2": r2, "n": n,
        "demand_controls": {"pop90": b[1]},
    }


# ---------------------------------------------------------------------------
# Unified dispatch: call these from post-estimation and counterfactual code
# ---------------------------------------------------------------------------

def second_stage(delta, price, raw, zm, zh, use_blp):
    """Run 2SLS and return a standardised result dict.

    Returns dict with keys: a0, a1, se_a0, se_a1, r2, demand_controls.
    demand_controls is None for the simple IV, dict for BLP.
    """
    if use_blp:
        blp = run_2sls_blp(delta, price, raw)
        return {
            "a0": blp["a0"], "a1": blp["a1"],
            "se_a0": blp["se_a0"], "se_a1": blp["se_a1"],
            "r2": blp["r2"],
            "demand_controls": blp["demand_controls"],
        }
    a0, a1, se0, se1, r2 = run_2sls(delta, price, zm, zh)
    return {
        "a0": a0, "a1": a1,
        "se_a0": se0, "se_a1": se1,
        "r2": r2,
        "demand_controls": None,
    }


def compute_xi(delta, price, a0, a1, demand_controls, bta_data):
    """Recover xi from the structural equation.

    delta_j = a0 + Z_j'gamma - a1*price_j + xi_j
    => xi_j = delta_j - a0 - Z_j'gamma + a1*price_j
    """
    xi = delta - a0 + a1 * price
    if demand_controls:
        for var, coeff in demand_controls.items():
            xi -= coeff * bta_data[var].to_numpy().astype(float)
    return xi
