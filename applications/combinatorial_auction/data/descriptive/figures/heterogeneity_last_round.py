"""Heterogeneity two-panel using last-round eligibility (capacity at exit)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from applications.combinatorial_auction.data.loaders import (
    load_raw, build_context, last_round_capacity,
)
from applications.combinatorial_auction.data.covariates import WEIGHT_ROUNDING_TICK
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax, pop_formatter,
)


def plot(raw, ctx):
    pop90 = raw["bta_data"]["pop90"].values
    # last-round capacity is in WEIGHT_ROUNDING_TICK units; scale back to pop
    elig = last_round_capacity(raw["bidder_data"]).astype(float) * WEIGHT_ROUNDING_TICK
    win = ctx["c_obs_bundles"].any(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): last-round eligibility distribution
    bins = np.logspace(np.log10(elig[elig > 0].min() * 0.9),
                       np.log10(elig.max() * 1.1), 25)
    ax1.hist(elig[~win], bins=bins, color=SLATE, alpha=0.40,
             label="Non-winners", edgecolor="white", linewidth=0.4, zorder=2)
    ax1.hist(elig[win],  bins=bins, color=NAVY,  alpha=0.85,
             label="Winners", edgecolor="white", linewidth=0.4, zorder=3)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(pop_formatter))
    ax1.set_xlabel("Eligibility (last active round)", fontsize=9, family="serif")
    ax1.set_ylabel("Number of bidders", fontsize=9, family="serif")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    style_ax(ax1)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    # Panel (b): assortative matching with last-round eligibility
    pkg_pop = ctx["c_obs_bundles"] @ pop90
    w = win & (pkg_pop > 0) & (elig > 0)
    log_e, log_p = np.log10(elig[w]), np.log10(pkg_pop[w])
    ax2.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)
    lo = min(log_e.min(), log_p.min()) - 0.3
    hi = max(log_e.max(), log_p.max()) + 0.3
    ax2.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1,
             zorder=2, label="eligibility constraint")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel(r"$\log_{10}$(eligibility, last round)", fontsize=9, family="serif")
    ax2.set_ylabel(r"$\log_{10}$(winning package population)", fontsize=9, family="serif")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_FIG / "fig_heterogeneity_last_round.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(log_e, log_p)[0, 1]
    print(f"  last-round elig (winners) {elig[win].min():,.0f}–{elig[win].max():,.0f}; "
          f"corr = {corr:.3f}; {(pkg_pop[w] > elig[w]).sum()}/{w.sum()} above constraint")


if __name__ == "__main__":
    raw = load_raw()
    plot(raw, build_context(raw))
