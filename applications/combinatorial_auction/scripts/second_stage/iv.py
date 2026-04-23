"""Second-stage IV regressions.

One entry point: `run_2sls(delta, raw, app)`. The regressors, instruments,
and sample filter come from `app.counterfactual`; presets in `_PRESETS` below.

All presets share regressors=[const, pop, neg_price] (pop = pop90_share) and
differ only in the excluded-instrument set used for neg_price (and pop, when
pop is treated as endogenous). The `far1500u_popexog` preset additionally
lists `pop` among the instruments, which makes pop exogenous and leaves only
neg_price endogenous.

Override any field via the `counterfactual:` block in the estimation YAML:

    counterfactual:
        iv: far1500u_popexog         # one of the keys in _PRESETS
        sample: all                  # 'all' | 'rural'
        pop_threshold: .inf          # rural cutoff
        distance_threshold: 500      # km, for legacy distant-mean IVs
        regressors:  [...]           # (optional; overrides preset)
        instruments: [...]

Returns `{a0, a1, se_a0, se_a1, r2, n, demand_controls}` where
    a0 = const coefficient (α₀)
    a1 = neg_price coefficient (the price sensitivity, α₂ in the paper)
    demand_controls = {raw_col_name: coef} for every non-(const, neg_price)
        regressor, e.g. {"pop90_share": α₁}. Consumed by `prepare_counterfactual`
        and `compute_xi`.
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

    # Uniform-weighted average of BLP_VARS over BTAs with d(j,k) > 1500 km.
    # pop90_share is treated as EXOGENOUS (listed in both regressors and
    # instruments); only neg_price is endogenous.
    "far1500u_popexog": dict(
        regressors  = _REGS_BLP,
        instruments = ["const", "pop"] + [f"farU1500_{v}" for v in BLP_VARS],
        sample="all",
    ),

    # Linear-ramp kernel: weight = 0 at d ≤ 1000 km, grows linearly to 1 at
    # d ≥ 2000 km, capped at 1 beyond.  Only two BLP variables are used as
    # excluded instruments (pop90 and imwl — the two level-scale quantities,
    # which carry most of the cross-j identifying variation). pop90_share is
    # treated as EXOGENOUS (listed in both regressors and instruments); only
    # neg_price is endogenous.
    "far_ramp1000_2000_popexog": dict(
        regressors  = _REGS_BLP,
        instruments = ["const", "pop", "ramp1000_2000_pop90", "ramp1000_2000_imwl"],
        sample="all",
    ),
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
    far_uniform_cuts = tuple(sorted({int(n.split("_", 1)[0][4:]) for n in instr_names
                                     if n.startswith("farU")}))
    # "ramp<lo>_<hi>_<var>"  →  (lo, hi) pairs
    ramp_pairs = tuple(sorted({
        (int(n.split("_")[0][4:]), int(n.split("_")[1]))
        for n in instr_names if n.startswith("ramp")
    }))
    return dict(
        needs_blp   = any(n.startswith("blp_")    for n in instr_names),
        needs_adj   = any(n.startswith("adj_")    for n in instr_names),
        needs_adjout= any(n.startswith("adjout_") for n in instr_names),
        out_radii   = out_radii,
        out_powers  = out_powers,
        far_uniform_cuts = far_uniform_cuts,
        ramp_pairs  = ramp_pairs,
    )


def _far_uniform(raw, min_km):
    """Uniform mean of BLP_VARS over BTAs with d(j,k) > D km.
    Columns ``farU<D>_<var>``."""
    df = _blp_df(raw)
    D = raw["geo_distance"].astype(float).copy()
    np.fill_diagonal(D, 0.0)
    mask = D > float(min_km)
    n_nbr = mask.sum(axis=1).astype(float)
    out, tag = {}, f"farU{int(min_km)}"
    for v in BLP_VARS:
        vals = df[v].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (mask @ vals) / n_nbr
        m[n_nbr == 0] = np.nan
        out[f"{tag}_{v}"] = m
    return out


def _far_ramp(raw, lo_km, hi_km):
    """Linear-ramp-weighted average of BLP_VARS: weight is 0 at d ≤ lo_km,
    1 at d ≥ hi_km, linear in between.  Columns ``ramp<lo>_<hi>_<var>``."""
    df = _blp_df(raw)
    D = raw["geo_distance"].astype(float).copy()
    np.fill_diagonal(D, 0.0)
    W = np.clip((D - float(lo_km)) / (float(hi_km) - float(lo_km)), 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    denom = W.sum(axis=1)
    out, tag = {}, f"ramp{int(lo_km)}_{int(hi_km)}"
    for v in BLP_VARS:
        vals = df[v].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (W @ vals) / denom
        m[denom == 0] = np.nan
        out[f"{tag}_{v}"] = m
    return out


def _columns(raw, pop_threshold, distance_threshold,
             needs_blp=False, needs_adj=False, needs_adjout=False,
             out_radii=(), out_powers=(), far_uniform_cuts=(), ramp_pairs=()):
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
    for Dk in far_uniform_cuts: cols.update(_far_uniform(raw, Dk))
    for (lo, hi) in ramp_pairs: cols.update(_far_ramp(raw, lo, hi))
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
