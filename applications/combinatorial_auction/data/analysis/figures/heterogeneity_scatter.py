"""Assortative matching: log(eligibility) vs log(winning package population)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.analysis.style import (
    NAVY, SLATE, DPI, style_ax,
)

OUT = Path(__file__).parent.parent / "output"


def plot(raw, ctx):
    pop90 = raw["bta_data"]["pop90"].values
    elig = raw["bidder_data"]["pops_eligible"].values
    c_obs = ctx["c_obs_bundles"]
    pkg_pop = c_obs @ pop90
    w = c_obs.any(axis=1) & (pkg_pop > 0)
    log_e, log_p = np.log(elig[w]), np.log(pkg_pop[w])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)

    lo = min(log_e.min(), log_p.min()) - 0.3
    hi = max(log_e.max(), log_p.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1, zorder=2,
            label="eligibility constraint")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("log(eligibility)", fontsize=9, family="serif")
    ax.set_ylabel("log(winning package population)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig_heterogeneity_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(log_e, log_p)[0, 1]
    n_binding = (pkg_pop[w] > elig[w] * 0.9).sum()
    print(f"  Correlation(log elig, log pkg pop): {corr:.3f}")
    print(f"  Winners within 10% of constraint: {n_binding}/{w.sum()}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot(raw, ctx)
