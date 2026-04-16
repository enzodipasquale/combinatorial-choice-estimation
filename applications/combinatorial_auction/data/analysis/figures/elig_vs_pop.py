"""Standalone scatter: log(initial eligibility) vs log(pkg pop), all bidders."""

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
    is_winner = c_obs.any(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(np.log(elig[~is_winner]),
               np.full((~is_winner).sum(), np.log(pop90.min()) - 0.8),
               s=15, alpha=0.35, color=SLATE, marker="|", zorder=2,
               label="Non-winners")

    w = is_winner & (pkg_pop > 0)
    ax.scatter(np.log(elig[w]), np.log(pkg_pop[w]),
               s=30, alpha=0.6, color=NAVY, zorder=3, label="Winners")

    lo = np.log(elig.min()) - 0.5
    hi = np.log(elig.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1, zorder=2,
            label="45° line")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("log(initial eligibility)", fontsize=9, family="serif")
    ax.set_ylabel("log(winning package population)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig_elig_vs_pop.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(np.log(elig[w]), np.log(pkg_pop[w]))[0, 1]
    print(f"  Total: {len(elig)}, winners: {is_winner.sum()}")
    print(f"  Correlation (winners, log-log): {corr:.3f}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot(raw, ctx)
