"""Eligibility distribution: winners vs non-winners (initial eligibility)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax, pop_formatter,
)

OUT = Path(__file__).parent.parent / "output"


def plot(raw, ctx):
    elig = raw["bidder_data"]["pops_eligible"].values
    is_winner = ctx["c_obs_bundles"].any(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    log_bins = np.logspace(np.log10(elig.min() * 0.9), np.log10(elig.max() * 1.1), 25)
    ax.hist(elig[~is_winner], bins=log_bins, color=SLATE, alpha=0.40,
            label="Non-winners", edgecolor="white", linewidth=0.4, zorder=2)
    ax.hist(elig[is_winner], bins=log_bins, color=NAVY, alpha=0.85,
            label="Winners", edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(pop_formatter))
    ax.set_xlabel("Eligibility", fontsize=9, family="serif")
    ax.set_ylabel("Number of bidders", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig_heterogeneity_hist.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  Bidders: {len(elig)} total, {is_winner.sum()} winners")
    print(f"  Eligibility range: {elig.min():,.0f} – {elig.max():,.0f}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot(raw, ctx)
