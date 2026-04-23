"""Reduced-form evidence of complementarities in the FCC C-block auction.

Two artifacts for the "Evidence of complementarities" slide:

    fig_complementarities.png   — log-log scatter of winning price vs own
                                   population across 480 BTAs + OLS fit.
    tab_complementarities.tex   — regressions of the OLS residuals on two
                                   spatial measures of nearby-BTA population:
                                     (a) within-MTA pop (sum of other BTAs
                                         sharing the same MTA);
                                     (b) gravity-weighted pop (Σ pop/dist²).
                                   Three columns: (a) alone, (b) alone, both.

Reads:
    bta_data       — pop90, bid
    aggregation_A  — BTA→MTA indicator matrix (from loaders)
    distance CSV   — 493×493 pair-wise distances in meters

Run:
    python -m applications.combinatorial_auction.data.descriptive.figures.complementarities
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from applications.combinatorial_auction.data.loaders import (
    load_raw, load_aggregation_matrix, RAW,
)
from applications.combinatorial_auction.data.descriptive import OUT_FIG, OUT_TAB
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax,
)

DIST_FILE = RAW / "geographic-distance-population-weighted-centroid.csv"


# ── small OLS with HC1 standard errors ────────────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    # HC1: White's heteroskedasticity-robust SEs, small-sample correction.
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(resid ** 2) @ X
    V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"b": b, "se": se, "n": n, "k": k,
            "r2": 1 - ss_res / ss_tot, "resid": resid}


# ── spatial features ─────────────────────────────────────────────────

def _within_mta_other_pop(pop: np.ndarray, A: np.ndarray) -> np.ndarray:
    """For each BTA j, sum of pop of OTHER BTAs sharing its MTA."""
    # A is (n_mta, n_bta); A.T A gives a (n_bta, n_bta) same-MTA indicator.
    same_mta = (A.T @ A).astype(bool)
    np.fill_diagonal(same_mta, False)
    return same_mta @ pop


def _gravity_pop(pop: np.ndarray, dist_km: np.ndarray) -> np.ndarray:
    """Σ_{k ≠ j} pop_k / dist(j,k)², with dist in km, self-term dropped."""
    d = dist_km.copy().astype(float)
    np.fill_diagonal(d, np.nan)
    d[~np.isfinite(d)] = np.nan
    d[d <= 0] = np.nan
    w = 1.0 / (d ** 2)
    # replace NaNs with 0 so they don't contribute
    w = np.where(np.isfinite(w), w, 0.0)
    return w @ pop


# ── artifacts ────────────────────────────────────────────────────────

def _load_distance_matrix(bta_ids: np.ndarray) -> np.ndarray:
    """Return pairwise distance matrix in km for the given continental bta_ids.

    The raw file is a 493×493 symmetric distance matrix indexed by BTA id
    (row i / col i = BTA i+1, including non-continental BTAs).  We subset
    by ``bta_ids - 1`` so the output is aligned with the ``bta_data``
    continental order used everywhere else."""
    raw = pd.read_csv(DIST_FILE, header=None).to_numpy()
    idx = np.asarray(bta_ids, dtype=int) - 1
    if raw.shape[0] < idx.max() + 1:
        raise ValueError(
            f"distance matrix {raw.shape} too small for max BTA id {idx.max() + 1}")
    sub = raw[np.ix_(idx, idx)]
    return sub / 1000.0   # m → km


def _fmt(x, dig=3):
    if not np.isfinite(x):
        return "--"
    return f"{x:.{dig}f}"


def _fmt_coef(b, se):
    return f"{_fmt(b)} & ({_fmt(se)})"


def build():
    raw = load_raw()
    bta = raw["bta_data"]
    bta_ids = bta["bta"].astype(int).to_numpy()
    pop = bta["pop90"].to_numpy(dtype=float)
    bid = bta["bid"].to_numpy(dtype=float)

    # --- First stage: log bid on log pop ---------------------------------
    keep = (pop > 0) & (bid > 0)
    lp, lb = np.log10(pop[keep]), np.log10(bid[keep])
    X1 = np.column_stack([np.ones(keep.sum()), lp])
    s1 = _ols(lb, X1)

    # --- Spatial features ------------------------------------------------
    A, _ = load_aggregation_matrix(bta_ids)
    D = _load_distance_matrix(bta_ids)

    wm_pop = _within_mta_other_pop(pop, A)
    gv_pop = _gravity_pop(pop, D)

    # Log-transform spatial features (with a floor for zeros) to match units
    # with log pop.  A BTA with no same-MTA siblings gets np.log10(1)=0.
    lwm = np.log10(np.maximum(wm_pop, 1))
    lgv = np.log10(np.maximum(gv_pop, np.percentile(gv_pop[gv_pop > 0], 1)))

    r = s1["resid"]
    lwm_k, lgv_k = lwm[keep], lgv[keep]

    # --- Second-stage regressions ---------------------------------------
    s_mta  = _ols(r, np.column_stack([np.ones_like(r), lwm_k]))
    s_grav = _ols(r, np.column_stack([np.ones_like(r), lgv_k]))
    s_both = _ols(r, np.column_stack([np.ones_like(r), lwm_k, lgv_k]))

    # --- Figure: scatter + fit ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lp, lb, s=18, alpha=0.55, color=NAVY,
               edgecolor="white", linewidth=0.3)
    xs = np.linspace(lp.min(), lp.max(), 100)
    ax.plot(xs, s1["b"][0] + s1["b"][1] * xs,
            color=SLATE, lw=1.2, ls="--",
            label=fr"OLS: $\beta={s1['b'][1]:.3f}$, $R^2={s1['r2']:.3f}$")
    ax.set_xlabel(r"$\log_{10}$(population)", fontsize=9, family="serif")
    ax.set_ylabel(r"$\log_{10}$(winning bid)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style_ax(ax)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("serif")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_complementarities.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Table: residual regressions ------------------------------------
    tex_lines = [
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r" & (1) & (2) & (3) \\",
        r" & MTA only & Gravity only & Both \\",
        r"\midrule",
        rf"$\log_{{10}}$(within-MTA pop) & {_fmt(s_mta['b'][1])}  & --  "
        rf"& {_fmt(s_both['b'][1])} \\",
        rf"                              & ({_fmt(s_mta['se'][1])}) & -- "
        rf"& ({_fmt(s_both['se'][1])}) \\[3pt]",
        rf"$\log_{{10}}$(gravity-pop)    & --  & {_fmt(s_grav['b'][1])} "
        rf"& {_fmt(s_both['b'][2])} \\",
        rf"                              & --  & ({_fmt(s_grav['se'][1])}) "
        rf"& ({_fmt(s_both['se'][2])}) \\",
        r"\midrule",
        rf"$R^2$ & {_fmt(s_mta['r2'])} & {_fmt(s_grav['r2'])} "
        rf"& {_fmt(s_both['r2'])} \\",
        rf"$N$   & {s_mta['n']:,} & {s_grav['n']:,} & {s_both['n']:,} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(tex_lines) + "\n"
    (OUT_TAB / "tab_complementarities.tex").write_text(tex)

    # --- Log summary ----------------------------------------------------
    print(f"  fig_complementarities: "
          f"stage-1 β={s1['b'][1]:.3f}, R²={s1['r2']:.3f}, N={s1['n']}")
    print(f"  tab_complementarities:")
    print(f"    (1) MTA only   : β={s_mta['b'][1]:.3f}  ({s_mta['se'][1]:.3f})")
    print(f"    (2) Gravity    : β={s_grav['b'][1]:.3f}  ({s_grav['se'][1]:.3f})")
    print(f"    (3) Both       : βmta={s_both['b'][1]:.3f}  ({s_both['se'][1]:.3f}),  "
          f"βgrav={s_both['b'][2]:.3f}  ({s_both['se'][2]:.3f})")


if __name__ == "__main__":
    build()
