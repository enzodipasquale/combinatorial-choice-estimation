"""Assortative matching: log(eligibility) vs log(winning package population)."""
import numpy as np
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI, style_ax


def plot(raw, ctx):
    pop90 = raw["bta_data"]["pop90"].values
    elig  = raw["bidder_data"]["pops_eligible"].values
    pkg_pop = ctx["c_obs_bundles"] @ pop90
    w = ctx["c_obs_bundles"].any(axis=1) & (pkg_pop > 0)
    log_e, log_p = np.log10(elig[w]), np.log10(pkg_pop[w])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)
    lo = min(log_e.min(), log_p.min()) - 0.3
    hi = max(log_e.max(), log_p.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color=SLATE, lw=1,
            zorder=2, label="eligibility constraint")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\log_{10}$(eligibility)", fontsize=9, family="serif")
    ax.set_ylabel(r"$\log_{10}$(winning package population)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_heterogeneity_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    n_bind = (pkg_pop[w] > elig[w] * 0.9).sum()
    print(f"  corr(log elig, log pkg pop) = {np.corrcoef(log_e, log_p)[0, 1]:.3f}; "
          f"{n_bind}/{w.sum()} within 10% of constraint")


if __name__ == "__main__":
    raw = load_raw()
    plot(raw, build_context(raw))
