"""Heterogeneity two-panel using last-round eligibility."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax, pop_formatter,
)
from applications.combinatorial_auction.data.descriptive.helpers import last_round_eligibility_pop

OUT = Path(__file__).parent.parent / "output"


def plot(raw, ctx):
    bidder = raw["bidder_data"]
    pop90 = raw["bta_data"]["pop90"].values
    c_obs = ctx["c_obs_bundles"]
    elig = last_round_eligibility_pop(bidder)
    is_winner = c_obs.any(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    log_bins = np.logspace(np.log10(elig[elig > 0].min() * 0.9),
                           np.log10(elig.max() * 1.1), 25)
    ax1.hist(elig[~is_winner], bins=log_bins, color=SLATE, alpha=0.40,
             label="Non-winners", edgecolor="white", linewidth=0.4, zorder=2)
    ax1.hist(elig[is_winner], bins=log_bins, color=NAVY, alpha=0.85,
             label="Winners", edgecolor="white", linewidth=0.4, zorder=3)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(pop_formatter))
    ax1.set_xlabel("Eligibility (last active round)", fontsize=9, family="serif")
    ax1.set_ylabel("Number of bidders", fontsize=9, family="serif")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    style_ax(ax1)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    pkg_pop = c_obs @ pop90
    w = is_winner & (pkg_pop > 0) & (elig > 0)
    log_e, log_p = np.log(elig[w]), np.log(pkg_pop[w])

    ax2.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)
    lo = min(log_e.min(), log_p.min()) - 0.3
    hi = max(log_e.max(), log_p.max()) + 0.3
    ax2.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1, zorder=2,
             label="eligibility constraint")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("log(eligibility, last round)", fontsize=9, family="serif")
    ax2.set_ylabel("log(winning package population)", fontsize=9, family="serif")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "fig_heterogeneity_last_round.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(log_e, log_p)[0, 1]
    n_above = (pkg_pop[w] > elig[w]).sum()
    print(f"  Last-round elig range (winners): {elig[is_winner].min():,.0f} – {elig[is_winner].max():,.0f}")
    print(f"  Correlation(log elig, log pkg pop): {corr:.3f}")
    print(f"  Winners above constraint: {n_above}/{w.sum()}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot(raw, ctx)
