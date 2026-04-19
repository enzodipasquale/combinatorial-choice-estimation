"""log(initial eligibility) vs log(winning package pop), all bidders."""
import numpy as np
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI, style_ax


def plot(raw, ctx):
    pop90 = raw["bta_data"]["pop90"].values
    elig  = raw["bidder_data"]["pops_eligible"].values
    pkg_pop = ctx["c_obs_bundles"] @ pop90
    win = ctx["c_obs_bundles"].any(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # non-winners as a rug below the winner cloud
    rug_y = np.log10(pop90.min()) - 0.8
    ax.scatter(np.log10(elig[~win]), np.full((~win).sum(), rug_y),
               s=15, alpha=0.35, color=SLATE, marker="|", zorder=2, label="Non-winners")

    w = win & (pkg_pop > 0)
    ax.scatter(np.log10(elig[w]), np.log10(pkg_pop[w]),
               s=30, alpha=0.6, color=NAVY, zorder=3, label="Winners")

    lo, hi = np.log10(elig.min()) - 0.5, np.log10(elig.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1, zorder=2, label="45° line")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\log_{10}$(initial eligibility)", fontsize=9, family="serif")
    ax.set_ylabel(r"$\log_{10}$(winning package population)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_elig_vs_pop.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    corr = np.corrcoef(np.log10(elig[w]), np.log10(pkg_pop[w]))[0, 1]
    print(f"  {len(elig)} bidders ({win.sum()} winners); "
          f"corr on winners = {corr:.3f}")


if __name__ == "__main__":
    raw = load_raw()
    plot(raw, build_context(raw))
