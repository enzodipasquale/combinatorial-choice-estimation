"""IV regressions for the second stage.

Two dispatch paths:
  simple  — δ ~ α0 − α1·price, instrumented by distant pop / hhinc means.
  blp     — δ ~ α0 + β·pop − α1·price, instrumented by (a) own percapin
            and (b) exclude-one-mean of BLP characteristics over each MTA.
            Restricted to the rural sample (pop < pop_threshold) to avoid
            the mis-specification of BLP IVs in very large BTAs.

`use_blp=True` ⇔ the estimation spec used pop-scaled errors (BLP demand IVs
with a pop control are identified only once errors vary by pop_j).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ...data.loaders import RAW  # raw-data directory

BLP_VARS = ("pop90", "percapin", "density", "hhinc35k", "grow9099", "imwl")


def _hc1_cov(X, resid):
    n, k = X.shape
    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXinv = np.linalg.pinv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    return (n / (n - k)) * XtXinv @ meat @ XtXinv


def _ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    cov = _hc1_cov(X, r)
    se = np.sqrt(np.abs(np.diag(cov)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2, r


def _tsls(X, y, Z):
    Pz = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(Pz, y, rcond=None)[0]
    r = y - X @ b
    cov = _hc1_cov(Pz, r)
    se = np.sqrt(np.abs(np.diag(cov)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2, r


# ── Simple IV instruments (distant means) ────────────────────────────

def simple_instruments(raw, distance_threshold=500):
    """For each BTA j, return (mean_pop90, std_pop90, mean_hhinc35k) over BTAs
    farther than `distance_threshold` km."""
    geo   = raw["geo_distance"]
    pop   = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    hh    = raw["bta_data"]["hhinc35k"].to_numpy(dtype=float)
    m = geo > distance_threshold
    zp_mean = np.array([pop[m[j]].mean() if m[j].any() else 0.0 for j in range(len(pop))])
    zp_std  = np.array([pop[m[j]].std()  if m[j].any() else 0.0 for j in range(len(pop))])
    zh_mean = np.array([hh[m[j]].mean()  if m[j].any() else 0.0 for j in range(len(pop))])
    return zp_mean, zp_std, zh_mean


def _run_simple(delta, price, zp_mean, zh_mean):
    n = len(delta)
    X = np.column_stack([np.ones(n), -price])
    Z = np.column_stack([zp_mean, zh_mean])
    b, se, r2, _ = _tsls(X, delta, Z)
    return {"a0": b[0], "a1": b[1],
            "se_a0": se[0], "se_a1": se[1],
            "r2": r2, "demand_controls": None}


# ── BLP instruments (exclude-one-mean within MTA) ────────────────────

def _blp_excludeone_within_mta(raw):
    """For each continental BTA, build the exclude-one-mean of BLP_VARS over
    BTAs in the same MTA. Returns {var: array of length n_btas}.

    Note: 'percapin' is included in the output dict as a *raw* per-BTA value
    (not an exclude-one-mean) so callers can use it as an own-BTA control.
    The current BLP IV does not use it — it's carried here in case a future
    spec does.
    """
    census = pd.read_csv(RAW / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    bta_mta = census[["BTA", "MTA"]].dropna().copy()
    bta_mta.columns = ["bta", "mta"]
    bta_mta["bta"] = bta_mta["bta"].astype(int)
    bta_mta["mta"] = bta_mta["mta"].astype(int)

    btadata  = pd.read_csv(RAW / "btadata_2004_03_12_1.csv")
    btastats = pd.read_csv(RAW / "btastatsexport.csv")
    df = (btadata[["bta", "pop90", "percapin", "hhinc35k", "density", "imwl"]]
          .merge(btastats[["bta", "grow9099"]], on="bta")
          .merge(bta_mta, on="bta"))

    bta_order = raw["bta_data"]["bta"].values
    df = df[df["bta"].isin(set(bta_order))].reset_index(drop=True)

    for v in BLP_VARS:
        s = df.groupby("mta")[v].transform("sum")
        c = df.groupby("mta")[v].transform("count")
        df[f"blp_{v}"] = (s - df[v]) / (c - 1)

    out = {}
    for col in [f"blp_{v}" for v in BLP_VARS] + ["percapin"]:
        d = dict(zip(df["bta"], df[col]))
        out[col] = np.array([d.get(b, np.nan) for b in bta_order])
    return out


def _run_blp(delta, price, raw, pop_threshold):
    """δ ~ α0 + β·pop − α1·price, BLP IVs on the rural sample."""
    pop = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    blp = _blp_excludeone_within_mta(raw)

    valid = (pop < pop_threshold)
    for v in BLP_VARS:
        valid &= ~np.isnan(blp[f"blp_{v}"])
    n = valid.sum()

    d, p = delta[valid], price[valid]
    pop_v = pop[valid]

    X = np.column_stack([np.ones(n), pop_v, -p])
    excl = np.column_stack([blp[f"blp_{v}"][valid] for v in BLP_VARS])
    Z = np.column_stack([np.ones(n), excl])
    b, se, r2, _ = _tsls(X, d, Z)

    return {"a0": b[0], "a1": b[2],
            "se_a0": se[0], "se_a1": se[2],
            "r2": r2, "n": n,
            "demand_controls": {"pop90": b[1]}}


# ── Dispatchers ──────────────────────────────────────────────────────

def second_stage(delta, price, raw, *, use_blp,
                 simple_instruments_cached=None,
                 pop_threshold=500_000):
    """Run the 2SLS appropriate for the spec.

    Returns dict(a0, a1, se_a0, se_a1, r2, demand_controls). `demand_controls`
    is None for simple IV, {"pop90": β_pop} for BLP.
    """
    if use_blp:
        return _run_blp(delta, price, raw, pop_threshold)
    if simple_instruments_cached is None:
        zp_mean, _, zh_mean = simple_instruments(raw)
    else:
        zp_mean, _, zh_mean = simple_instruments_cached
    return _run_simple(delta, price, zp_mean, zh_mean)


def compute_xi(delta, price, a0, a1, demand_controls, bta_data):
    """ξ_j = δ_j − α0 + α1·p_j − Z_j'γ."""
    xi = delta - a0 + a1 * price
    if demand_controls:
        for var, coef in demand_controls.items():
            xi -= coef * bta_data[var].to_numpy(dtype=float)
    return xi
