"""Second-stage IV regressions.

One entry point, `run_2sls(delta, raw, app)`. The regressors, instruments, and
sample filter come from `app.counterfactual` in the estimation config; sensible
defaults are resolved from presets keyed on `error_scaling`:

    error_scaling: pop      → preset 'blp'     (δ ~ const + pop − α₁·price,
                                                 BLP exclude-one-mean IVs,
                                                 rural sample)
    otherwise               → preset 'simple'  (δ ~ const − α₁·price,
                                                 distant-mean IVs, all BTAs)

Any preset field can be overridden in the config:

    counterfactual:
        iv: blp                       # 'simple' | 'blp' (optional)
        regressors:   [const, pop, neg_price]
        instruments:  [const, blp_pop90, blp_percapin, ...]
        sample:       rural           # 'all' | 'rural'
        pop_threshold: 500_000        # rural cutoff
        distance_threshold: 500       # km, for distant-mean IVs

Output: dict(a0, a1, se_a0, se_a1, r2, n, demand_controls), where
`demand_controls` maps each non-(const, neg_price) regressor to its raw
BTA-level column name and coefficient (consumed by prepare_counterfactual
and compute_xi).
"""
import numpy as np
import pandas as pd
from ...data.loaders import RAW

BLP_VARS = ("pop90", "percapin", "density", "hhinc35k", "grow9099", "imwl")

# Regressor name → raw BTA column (so compute_xi can subtract Z'γ using raw data).
_REGRESSOR_RAW_COL = {"pop": "pop90"}

_PRESETS = {
    "simple": {
        "regressors":  ["const", "neg_price"],
        "instruments": ["const", "distant_pop_mean", "distant_hhinc_mean"],
        "sample":      "all",
    },
    "blp": {
        "regressors":  ["const", "pop", "neg_price"],
        "instruments": ["const"] + [f"blp_{v}" for v in BLP_VARS],
        "sample":      "rural",
    },
}


# ── Linear algebra primitives ────────────────────────────────────────

def _hc1_cov(X, r):
    n, k = X.shape
    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtXinv = np.linalg.pinv(X.T @ X)
    meat = (X * r[:, None]).T @ (X * r[:, None])
    return (n / (n - k)) * XtXinv @ meat @ XtXinv


def _tsls(X, y, Z):
    Pz = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b  = np.linalg.lstsq(Pz, y, rcond=None)[0]
    r  = y - X @ b
    se = np.sqrt(np.abs(np.diag(_hc1_cov(Pz, r))))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


# ── Column registry (named regressors and instruments) ──────────────

def _distant_means(raw, threshold_km):
    geo  = raw["geo_distance"]
    pop  = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    hh   = raw["bta_data"]["hhinc35k"].to_numpy(dtype=float)
    m    = geo > threshold_km
    def _mean(col):
        return np.array([col[m[j]].mean() if m[j].any() else 0.0 for j in range(len(col))])
    return _mean(pop), _mean(hh)


def _blp_excludeone_within_mta(raw):
    """Per-BTA exclude-one-mean of BLP_VARS over BTAs in the same MTA, plus
    the raw own-BTA `percapin` for optional use as a control."""
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
    df = df[df["bta"].isin(set(raw["bta_data"]["bta"].values))].reset_index(drop=True)

    for v in BLP_VARS:
        s = df.groupby("mta")[v].transform("sum")
        c = df.groupby("mta")[v].transform("count")
        df[f"blp_{v}"] = (s - df[v]) / (c - 1)

    order = raw["bta_data"]["bta"].values
    out = {}
    for col in [f"blp_{v}" for v in BLP_VARS] + ["percapin"]:
        d = dict(zip(df["bta"], df[col]))
        out[col] = np.array([d.get(b, np.nan) for b in order])
    return out


def _columns(raw, pop_threshold, distance_threshold):
    """Build every named array that the 2SLS may reference, plus the rural mask."""
    pop   = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    dpop, dhh = _distant_means(raw, distance_threshold)
    blp = _blp_excludeone_within_mta(raw)
    cols = {
        "const": np.ones_like(pop), "price": price, "neg_price": -price,
        "pop": pop, "percapin": blp["percapin"],
        "distant_pop_mean":   dpop,
        "distant_hhinc_mean": dhh,
    }
    cols.update({f"blp_{v}": blp[f"blp_{v}"] for v in BLP_VARS})
    return cols, (pop < pop_threshold)


# ── Resolve config into concrete IV spec ────────────────────────────

def _resolve(app):
    cf = app.get("counterfactual") or {}
    iv_name = cf.get("iv") or ("blp" if app.get("error_scaling") == "pop" else "simple")
    if iv_name not in _PRESETS:
        raise ValueError(f"counterfactual.iv must be 'simple' or 'blp', got {iv_name!r}")
    p = _PRESETS[iv_name]
    return dict(
        regressors         = cf.get("regressors",         p["regressors"]),
        instruments        = cf.get("instruments",        p["instruments"]),
        sample             = cf.get("sample",             p["sample"]),
        pop_threshold      = cf.get("pop_threshold",      500_000),
        distance_threshold = cf.get("distance_threshold", 500),
    )


# ── Main entry point ────────────────────────────────────────────────

def run_2sls(delta, raw, app):
    """Run the 2SLS implied by app.counterfactual (or its defaults).

    δ must align with raw['bta_data'] (length n_btas, continental).
    """
    opts = _resolve(app)
    cols, rural = _columns(raw, opts["pop_threshold"], opts["distance_threshold"])

    regs, instrs = opts["regressors"], opts["instruments"]
    if "const" not in regs or "neg_price" not in regs:
        raise ValueError(f"regressors must include 'const' and 'neg_price', got {regs}")

    mask = rural.copy() if opts["sample"] == "rural" else np.ones(len(delta), bool)
    for name in set(regs) | set(instrs):
        mask &= ~np.isnan(cols[name])

    X = np.column_stack([cols[r][mask] for r in regs])
    Z = np.column_stack([cols[i][mask] for i in instrs])
    b, se, r2 = _tsls(X, delta[mask], Z)

    i_const, i_price = regs.index("const"), regs.index("neg_price")
    demand_controls = {
        _REGRESSOR_RAW_COL.get(regs[i], regs[i]): float(b[i])
        for i in range(len(regs)) if i not in (i_const, i_price)
    }
    return {
        "a0": float(b[i_const]), "a1": float(b[i_price]),
        "se_a0": float(se[i_const]), "se_a1": float(se[i_price]),
        "r2": float(r2), "n": int(mask.sum()),
        "demand_controls": demand_controls,
    }


def compute_xi(delta, price, a0, a1, demand_controls, bta_data):
    """ξ_j = δ_j − α₀ + α₁·p_j − Z_j'γ."""
    xi = delta - a0 + a1 * price
    for var, coef in (demand_controls or {}).items():
        xi -= coef * bta_data[var].to_numpy(dtype=float)
    return xi
