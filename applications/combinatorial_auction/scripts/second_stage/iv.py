"""Second-stage IV regressions.

One entry point: `run_2sls(delta, raw, app)`. The regressors, instruments,
and sample filter come from `app.counterfactual`; sensible defaults apply
when the block is absent.

Two presets are supported:

    simple : regressors=[const, neg_price]
             instruments=[const, distant_pop_mean, distant_hhinc_mean]
             sample=all

    blp    : regressors=[const, pop, neg_price]        (pop = pop90_share)
             instruments=[const] + BLP within-MTA exclude-one means
             sample=rural

Override any field via the `counterfactual:` block in the estimation YAML:

    counterfactual:
        iv: simple                   # 'simple' | 'blp'
        sample: all                  # 'all' | 'rural'
        pop_threshold: .inf          # rural cutoff
        distance_threshold: 500      # km, for distant-mean IVs
        regressors:  [const, neg_price]    # (optional; overrides preset)
        instruments: [const, distant_pop_mean]

Returns `{a0, a1, se_a0, se_a1, r2, n, demand_controls}` where
`demand_controls` maps each non-(const, neg_price) regressor to its
raw BTA-level column name and estimated coefficient (consumed by
`prepare_counterfactual` and `compute_xi`).
"""
import numpy as np
import pandas as pd

from ...data.loaders import RAW

BLP_VARS = ("pop90", "percapin", "density", "hhinc35k", "grow9099", "imwl")

# Map each non-const / non-neg_price regressor to its raw BTA column name so
# `compute_xi` can subtract Z'γ using raw data.  Share-normalized `pop90_share`
# matches the `pop` regressor so γ is on a unit-free scale.
_REGRESSOR_RAW_COL = {"pop": "pop90_share"}

def _ivs(prefix):
    return ["const"] + [f"{prefix}_{v}" for v in BLP_VARS]


# Every preset uses regressors=[const, pop, neg_price]; they differ only in
# which BLP_VARS aggregator is used as the excluded instrument set.
_REGS_BLP = ["const", "pop", "neg_price"]

_PRESETS = {
    # Alias kept for back-compat: same as 'blp' below.
    "simple":      dict(regressors=_REGS_BLP, instruments=_ivs("blp"),   sample="all"),

    # Within-MTA exclude-one mean of BLP_VARS (classical BLP instrument).
    "blp":         dict(regressors=_REGS_BLP, instruments=_ivs("blp"),   sample="all"),

    # Mean over BTAs adjacent to j AND in the same MTA as j.
    "adj_in":      dict(regressors=_REGS_BLP, instruments=_ivs("adj"),   sample="all"),

    # Mean over BTAs adjacent to j AND in a different MTA (interior BTAs of
    # large MTAs get NaN → dropped from sample).
    "adj_out":     dict(regressors=_REGS_BLP, instruments=_ivs("adjout"),sample="all"),

    # Inverse-distance-weighted means over all cross-MTA BTAs.  w_k = 1/d_k^p.
    "out_w1":      dict(regressors=_REGS_BLP, instruments=_ivs("outW1"), sample="all"),
    "out_w2":      dict(regressors=_REGS_BLP, instruments=_ivs("outW2"), sample="all"),
    "out_w4":      dict(regressors=_REGS_BLP, instruments=_ivs("outW4"), sample="all"),
}


# ── Linear algebra primitives ─────────────────────────────────────────

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


# ── Column registry (named regressors and instruments) ───────────────

def _distant_means(raw, threshold_km):
    geo  = raw["geo_distance"]
    pop  = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    hh   = raw["bta_data"]["hhinc35k"].to_numpy(dtype=float)
    m    = geo > threshold_km
    def _mean(col):
        return np.array([col[m[j]].mean() if m[j].any() else 0.0
                         for j in range(len(col))])
    return _mean(pop), _mean(hh)


def _blp_df(raw):
    """Merged BTA covariates + MTA crosswalk, in the order of raw['bta_data']."""
    census = pd.read_csv(RAW / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    bta_mta = (census[["BTA", "MTA"]].dropna().copy()
               .rename(columns={"BTA": "bta", "MTA": "mta"}))
    bta_mta[["bta", "mta"]] = bta_mta[["bta", "mta"]].astype(int)

    btadata  = pd.read_csv(RAW / "btadata_2004_03_12_1.csv")
    btastats = pd.read_csv(RAW / "btastatsexport.csv")
    df = (btadata[["bta", "pop90", "percapin", "hhinc35k", "density", "imwl"]]
          .merge(btastats[["bta", "grow9099"]], on="bta")
          .merge(bta_mta, on="bta"))
    order = raw["bta_data"]["bta"].values
    df = df.set_index("bta").reindex(order).reset_index()
    return df


def _blp_excludeone_within_mta(raw):
    """Per-BTA exclude-one-mean of BLP_VARS over OTHER BTAs in the same MTA."""
    df = _blp_df(raw)
    out = {}
    for v in BLP_VARS:
        s = df.groupby("mta")[v].transform("sum")
        c = df.groupby("mta")[v].transform("count")
        out[f"blp_{v}"] = ((s - df[v]) / (c - 1)).to_numpy(dtype=float)
    return out


def _adj_mean(raw, same_mta_only):
    """Per-BTA mean of BLP_VARS over BTAs that are adjacent to j and either
    (same_mta_only=True) in the same MTA as j, or (False) OUTSIDE j's MTA.
    NaN if j has no matching neighbor.
    """
    df = _blp_df(raw)
    adj = raw["bta_adjacency"].astype(bool)
    mta = df["mta"].to_numpy()
    same_mta = (mta[:, None] == mta[None, :])
    np.fill_diagonal(same_mta, False)
    nbr = adj & (same_mta if same_mta_only else ~same_mta)

    n_nbr = nbr.sum(axis=1).astype(float)
    out, prefix = {}, "adj" if same_mta_only else "adjout"
    for v in BLP_VARS:
        vals = df[v].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (nbr @ vals) / n_nbr
        mean[n_nbr == 0] = np.nan
        out[f"{prefix}_{v}"] = mean
    return out


def _adj_excludeone_within_mta(raw):
    return _adj_mean(raw, same_mta_only=True)


def _adj_excludeone_outside_mta(raw):
    return _adj_mean(raw, same_mta_only=False)


def _out_radius(raw, radius_km):
    """Mean of BLP_VARS over BTAs with mta(k) ≠ mta(j) and dist(j,k) ≤ R.
    Produces columns ``outR<R>_<var>``.  NaN if no qualifying neighbor."""
    df = _blp_df(raw)
    D   = raw["geo_distance"]
    mta = df["mta"].to_numpy()
    diff_mta = (mta[:, None] != mta[None, :])
    within   = (D <= float(radius_km))
    np.fill_diagonal(within, False)
    nbr = diff_mta & within
    n_nbr = nbr.sum(axis=1).astype(float)
    out = {}
    tag = f"outR{int(radius_km)}"
    for v in BLP_VARS:
        vals = df[v].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (nbr @ vals) / n_nbr
        m[n_nbr == 0] = np.nan
        out[f"{tag}_{v}"] = m
    return out


def _out_invdist(raw, power):
    """Inverse-distance-weighted mean over BTAs with mta(k) ≠ mta(j).
    Columns ``outW<power>_<var>``."""
    df = _blp_df(raw)
    D   = raw["geo_distance"].astype(float).copy()
    np.fill_diagonal(D, np.inf)         # kill self-weight
    mta = df["mta"].to_numpy()
    mask = (mta[:, None] != mta[None, :])
    W = np.where(mask, 1.0 / np.where(D > 0, D, np.inf) ** float(power), 0.0)
    denom = W.sum(axis=1)
    out, tag = {}, f"outW{int(power)}"
    for v in BLP_VARS:
        vals = df[v].to_numpy(dtype=float)
        num = W @ vals
        with np.errstate(invalid="ignore", divide="ignore"):
            m = num / denom
        m[denom == 0] = np.nan
        out[f"{tag}_{v}"] = m
    return out


def _iv_requirements(instr_names):
    """Inspect an instrument list and return the keyword-arg bundle needed by
    ``_columns`` to populate exactly the required instrument families."""
    out_radii  = tuple(sorted({int(n.split("_", 1)[0][4:]) for n in instr_names
                               if n.startswith("outR")}))
    out_powers = tuple(sorted({int(n.split("_", 1)[0][4:]) for n in instr_names
                               if n.startswith("outW")}))
    return dict(
        needs_blp   = any(n.startswith("blp_")    for n in instr_names),
        needs_adj   = any(n.startswith("adj_")    for n in instr_names),
        needs_adjout= any(n.startswith("adjout_") for n in instr_names),
        out_radii   = out_radii,
        out_powers  = out_powers,
    )


def _columns(raw, pop_threshold, distance_threshold,
             needs_blp=False, needs_adj=False, needs_adjout=False,
             out_radii=(), out_powers=()):
    pop       = raw["bta_data"]["pop90"].to_numpy(dtype=float)
    pop_share = raw["bta_data"]["pop90_share"].to_numpy(dtype=float)
    price     = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    dpop, dhh = _distant_means(raw, distance_threshold)
    cols = {
        "const": np.ones_like(pop), "neg_price": -price,
        "pop": pop_share,
        "distant_pop_mean":   dpop,
        "distant_hhinc_mean": dhh,
    }
    if needs_blp:    cols.update(_blp_excludeone_within_mta(raw))
    if needs_adj:    cols.update(_adj_excludeone_within_mta(raw))
    if needs_adjout: cols.update(_adj_excludeone_outside_mta(raw))
    for R in out_radii: cols.update(_out_radius(raw, R))
    for p in out_powers: cols.update(_out_invdist(raw, p))
    return cols, (pop < pop_threshold)


# ── Resolve config into concrete IV spec ─────────────────────────────

def _resolve(app):
    cf = app.get("counterfactual") or {}
    iv_name = cf.get("iv") or ("blp" if app.get("error_scaling") == "pop" else "simple")
    if iv_name not in _PRESETS:
        raise ValueError(f"counterfactual.iv must be one of {list(_PRESETS)}, got {iv_name!r}")
    p = _PRESETS[iv_name]
    return dict(
        regressors         = cf.get("regressors",         p["regressors"]),
        instruments        = cf.get("instruments",        p["instruments"]),
        sample             = cf.get("sample",             p["sample"]),
        pop_threshold      = cf.get("pop_threshold",      500_000),
        distance_threshold = cf.get("distance_threshold", 500),
    )


# ── Main entry point ─────────────────────────────────────────────────

def run_2sls(delta, raw, app):
    """Run the 2SLS implied by `app.counterfactual` (or its defaults).

    δ must align with `raw['bta_data']` (length n_btas, continental).
    """
    opts = _resolve(app)
    regs, instrs = opts["regressors"], opts["instruments"]
    cols, rural = _columns(raw, opts["pop_threshold"], opts["distance_threshold"],
                           **_iv_requirements(instrs))
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
