"""Reduced-form evidence of complementarities in the FCC C-block auction.

One hedonic regression across the 480 continental BTAs,

    log10(bid_j) = alpha + beta * log10(pop_j)
                 + gamma1 * log10(within-MTA pop_j)
                 + gamma2 * log10(gravity-pop_j)
                 + eps_j,

where
    within-MTA pop_j = sum_{j' in MTA(j), j' != j} pop_{j'}
    gravity-pop_j    = sum_{j' != j} pop_{j'} / d_tilde^4,
                       d_tilde = d_{jj'} / 1000  (distance in thousands of km)

Reported with HC1 standard errors, in four nested specifications (own pop
only / + within-MTA / + gravity / + both).

Artifacts:
    fig_complementarities.png   stage-1 scatter + OLS fit (log price on log pop)
    tab_complementarities.tex   nested regressions

Run:
    python -m applications.combinatorial_auction.data.descriptive.figures.complementarities
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import (
    load_raw, load_aggregation_matrix, RAW,
)
from applications.combinatorial_auction.data.descriptive import OUT_FIG, OUT_TAB
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax,
)

DIST_FILE = RAW / "geographic-distance-population-weighted-centroid.csv"
ADJ_FILE  = RAW / "adjacency-bta.csv"


# ── small OLS with HC1 robust standard errors ─────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * (resid[:, None] ** 2)).T @ X
    V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"b": b, "se": se, "n": n, "k": k,
            "r2": 1 - ss_res / ss_tot, "resid": resid}


# ── spatial features ─────────────────────────────────────────────────

def _load_square_matrix(path, bta_ids, *, fill_diag=None):
    """Load the 493×493 Fox-Bajari BTA matrix and subset to the continental
    order in ``bta_ids`` (rows/cols i correspond to BTA id i+1)."""
    raw = pd.read_csv(path, header=None).to_numpy(dtype=float)
    idx = np.asarray(bta_ids, dtype=int) - 1
    if raw.shape[0] < idx.max() + 1:
        raise ValueError(
            f"{path.name} shape {raw.shape} too small for max BTA id {idx.max() + 1}")
    sub = raw[np.ix_(idx, idx)]
    if fill_diag is not None:
        np.fill_diagonal(sub, fill_diag)
    return sub


def _adjacency_pop(pop, adj):
    """adjacency_j = sum of pop over BTAs sharing a boundary with j."""
    A = (adj > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A @ pop


def _within_mta_other_pop(pop, A):
    """For each BTA j, sum of pop of OTHER BTAs sharing its MTA.

    ``A`` is the (n_mta, n_bta) indicator matrix from load_aggregation_matrix;
    ``A.T @ A`` gives a (n_bta, n_bta) same-MTA indicator that we zero on
    the diagonal before summing."""
    same = (A.T @ A).astype(bool)
    np.fill_diagonal(same, False)
    return same @ pop


def _gravity_pop(pop, d_km, *, power: int = 4):
    """gravity_j = sum_{j' != j} pop_{j'} / d_tilde^power,
    with d_tilde = d_km / 1000 (distance in thousands of km)."""
    d = d_km.copy().astype(float)
    np.fill_diagonal(d, np.nan)
    d[~np.isfinite(d)] = np.nan
    d[d <= 0] = np.nan
    d_tilde = d / 1000.0
    w = 1.0 / (d_tilde ** power)
    w = np.where(np.isfinite(w), w, 0.0)
    return w @ pop


# ── helpers ──────────────────────────────────────────────────────────

def _fmt(x, dig=3):
    if not np.isfinite(x):
        return "--"
    return f"{x:.{dig}f}"


# ── main ─────────────────────────────────────────────────────────────

def build():
    raw = load_raw()
    bta = raw["bta_data"]
    bta_ids = bta["bta"].astype(int).to_numpy()
    pop = bta["pop90"].to_numpy(dtype=float)
    bid = bta["bid"].to_numpy(dtype=float)

    # Log10 of own pop / bid.
    keep = (pop > 0) & (bid > 0)
    lp = np.log10(pop[keep])
    lb = np.log10(bid[keep])
    ones = np.ones_like(lb)

    # Spatial matrices + features.
    d_km = _load_square_matrix(DIST_FILE, bta_ids) / 1000.0     # m → km
    A, _ = load_aggregation_matrix(bta_ids)
    mta_pop = _within_mta_other_pop(pop, A)
    gvy4    = _gravity_pop(pop, d_km, power=4)
    l_mta = np.log10(np.maximum(mta_pop, 1))[keep]
    l_g4  = np.log10(np.maximum(gvy4, 1))[keep]

    # Nested specifications.  (1) own pop only ; (2) + within-MTA ;
    # (3) + gravity ; (4) both complementarity regressors.
    s1 = _ols(lb, np.column_stack([ones, lp]))
    s2 = _ols(lb, np.column_stack([ones, lp, l_mta]))
    s3 = _ols(lb, np.column_stack([ones, lp, l_g4]))
    s4 = _ols(lb, np.column_stack([ones, lp, l_mta, l_g4]))

    # ── figure: log price vs log pop (column-1 fit) ──────────────────
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

    # ── table: stage-2 coefficients ──────────────────────────────────
    # Table: single-column hedonic with all three regressors.
    tex_lines = [
        r"\begin{tabular}{l c}",
        r"\toprule",
        r" & Estimate (s.e.) \\",
        r"\midrule",
        rf"$\log_{{10}}$(pop)              "
        rf"& {_fmt(s4['b'][1])}  ({_fmt(s4['se'][1])}) \\",
        rf"$\log_{{10}}$(within-MTA pop)   "
        rf"& {_fmt(s4['b'][2])}  ({_fmt(s4['se'][2])}) \\",
        rf"$\log_{{10}}$(gravity, $d^{{4}}$) "
        rf"& {_fmt(s4['b'][3])}  ({_fmt(s4['se'][3])}) \\",
        r"\midrule",
        rf"$R^2$ & {_fmt(s4['r2'])} \\",
        rf"$N$   & {s4['n']:,} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT_TAB / "tab_complementarities.tex").write_text("\n".join(tex_lines) + "\n")

    # ── log ──────────────────────────────────────────────────────────
    print(f"  fig_complementarities: stage-1 β={s1['b'][1]:.3f}, "
          f"R²={s1['r2']:.3f}, N={s1['n']}")
    print("  tab_complementarities (HC1 SEs; nested hedonic):")
    for tag, s in [("(1) pop",              s1),
                   ("(2) pop + mta",        s2),
                   ("(3) pop + grav(d⁴)",   s3),
                   ("(4) pop + mta + grav", s4)]:
        coefs = "  ".join(f"β{i}={s['b'][i]:+.3f}({s['se'][i]:.3f})"
                          for i in range(1, len(s["b"])))
        print(f"    {tag:<22}  R²={s['r2']:.3f}   {coefs}")
    print(f"\n  corr(log within-MTA, log gravity d⁴) = "
          f"{np.corrcoef(l_mta, l_g4)[0, 1]:+.3f}")


if __name__ == "__main__":
    build()
