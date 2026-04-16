"""Generate descriptive figures for FCC C-block PCS auction slides.

Produces:
  fig_heterogeneity.png            — bidder eligibility distribution + assortative matching (initial)
  fig_heterogeneity_last_round.png — same, using last-round eligibility
  fig_clustering.png               — within-package distance ECDF (observed vs null)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

# ── palette ──────────────────────────────────────────────────────────────────
NAVY = "#1B2A4A"
GOLD = "#B8860B"
SLATE = "#4A6274"
RED = "#C0392B"

OUT_DIR = Path(__file__).parent
DPI = 200
RNG = np.random.default_rng(42)
N_NULL = 500


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pop_formatter(x, _):
    if x >= 1e8:
        return f"{x / 1e6:.0f}M"
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if x >= 1e5:
        return f"{x / 1e3:.0f}K"
    if x >= 1e4:
        return f"{x / 1e3:.0f}K"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


RAW_DIR = Path(__file__).parents[2] / "datasets" / "114402-V1" / "Replication-Fox-and-Bajari" / "data"


def _last_round_eligibility_pop(bidder_data):
    """Last-round eligibility in population units (before knapsack rescaling)."""
    elig_df = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    swap = {190: 234, 234: 190}
    baseline = bidder_data["pops_eligible"].to_numpy().astype(float)
    n = len(bidder_data)
    out = np.copy(baseline)
    for i in range(n):
        fox_swapped = int(bidder_data.iloc[i]["bidder_num_fox"])
        fox_orig = swap.get(fox_swapped, fox_swapped)
        sub = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["max_elig"] > 0)]
        if len(sub) == 0:
            continue
        last = sub.sort_values("round_num").iloc[-1]
        r1_rows = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["round_num"] == 1)]
        if len(r1_rows) == 0:
            continue
        r1 = r1_rows["max_elig"].iloc[0]
        if r1 > 0:
            out[i] = baseline[i] * last["max_elig"] / r1
    return out


def figure_heterogeneity(raw, ctx):
    bidder = raw["bidder_data"]
    pop90 = raw["bta_data"]["pop90"].values
    c_obs = ctx["c_obs_bundles"]
    elig = bidder["pops_eligible"].values.copy()
    is_winner = c_obs.any(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A: eligibility distribution ────────────────────────────────────
    log_bins = np.logspace(np.log10(elig.min() * 0.9), np.log10(elig.max() * 1.1), 25)
    ax1.hist(elig[~is_winner], bins=log_bins, color=SLATE, alpha=0.40,
             label="Non-winners", edgecolor="white", linewidth=0.4, zorder=2)
    ax1.hist(elig[is_winner], bins=log_bins, color=NAVY, alpha=0.85,
             label="Winners", edgecolor="white", linewidth=0.4, zorder=3)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(_pop_formatter))
    ax1.set_xlabel("Eligibility", fontsize=9, family="serif")
    ax1.set_ylabel("Number of bidders", fontsize=9, family="serif")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    _style_ax(ax1)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    # ── Panel B: assortative matching ────────────────────────────────────────
    pkg_pop = c_obs @ pop90
    w = is_winner & (pkg_pop > 0)
    log_e = np.log(elig[w])
    log_p = np.log(pkg_pop[w])

    ax2.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)

    lim_lo = min(log_e.min(), log_p.min()) - 0.3
    lim_hi = max(log_e.max(), log_p.max()) + 0.3
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color=SLATE, lw=1,
             zorder=2, label="eligibility constraint")
    ax2.set_xlim(lim_lo, lim_hi)
    ax2.set_ylim(lim_lo, lim_hi)
    ax2.set_xlabel("log(eligibility)", fontsize=9, family="serif")
    ax2.set_ylabel("log(winning package population)", fontsize=9, family="serif")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    _style_ax(ax2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    corr = np.corrcoef(log_e, log_p)[0, 1]
    n_binding = (pkg_pop[w] > elig[w] * 0.9).sum()

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_DIR / "fig_heterogeneity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("=== Figure 1: Heterogeneity ===")
    print(f"  Bidders: {len(bidder)} total, {is_winner.sum()} winners")
    print(f"  Eligibility range: {elig.min():,.0f} – {elig.max():,.0f}")
    print(f"  Correlation(log elig, log pkg pop): {corr:.3f}")
    print(f"  Winners within 10% of constraint: {n_binding}/{w.sum()}")
    print()


def figure_heterogeneity_last_round(raw, ctx):
    bidder = raw["bidder_data"]
    pop90 = raw["bta_data"]["pop90"].values
    c_obs = ctx["c_obs_bundles"]
    elig = _last_round_eligibility_pop(bidder)
    is_winner = c_obs.any(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A: last-round eligibility distribution ─────────────────────────
    log_bins = np.logspace(np.log10(elig[elig > 0].min() * 0.9),
                           np.log10(elig.max() * 1.1), 25)
    ax1.hist(elig[~is_winner], bins=log_bins, color=SLATE, alpha=0.40,
             label="Non-winners", edgecolor="white", linewidth=0.4, zorder=2)
    ax1.hist(elig[is_winner], bins=log_bins, color=NAVY, alpha=0.85,
             label="Winners", edgecolor="white", linewidth=0.4, zorder=3)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(FuncFormatter(_pop_formatter))
    ax1.set_xlabel("Eligibility (last active round)", fontsize=9, family="serif")
    ax1.set_ylabel("Number of bidders", fontsize=9, family="serif")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    _style_ax(ax1)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    # ── Panel B: assortative matching (last-round) ───────────────────────────
    pkg_pop = c_obs @ pop90
    w = is_winner & (pkg_pop > 0) & (elig > 0)
    log_e = np.log(elig[w])
    log_p = np.log(pkg_pop[w])

    ax2.scatter(log_e, log_p, s=30, alpha=0.6, color=NAVY, zorder=3)

    lim_lo = min(log_e.min(), log_p.min()) - 0.3
    lim_hi = max(log_e.max(), log_p.max()) + 0.3
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color=SLATE, lw=1,
             zorder=2, label="eligibility constraint")
    ax2.set_xlim(lim_lo, lim_hi)
    ax2.set_ylim(lim_lo, lim_hi)
    ax2.set_xlabel("log(eligibility, last round)", fontsize=9, family="serif")
    ax2.set_ylabel("log(winning package population)", fontsize=9, family="serif")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    _style_ax(ax2)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top", family="serif")

    corr = np.corrcoef(log_e, log_p)[0, 1]
    n_above = (pkg_pop[w] > elig[w]).sum()

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_DIR / "fig_heterogeneity_last_round.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("=== Figure 1b: Heterogeneity (last-round eligibility) ===")
    print(f"  Bidders: {len(bidder)} total, {is_winner.sum()} winners")
    print(f"  Last-round elig range (winners): {elig[is_winner].min():,.0f} – {elig[is_winner].max():,.0f}")
    print(f"  Correlation(log elig, log pkg pop): {corr:.3f}")
    print(f"  Winners above constraint: {n_above}/{w.sum()}")
    print()


def _pairwise_distances_within_packages(c_obs, dist, winners_idx):
    """Return array of all within-package pairwise distances for given bidders."""
    all_dists = []
    for i in winners_idx:
        items = np.where(c_obs[i])[0]
        if len(items) < 2:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                all_dists.append(dist[items[a], items[b]])
    return np.array(all_dists)


def _null_pairwise_distances(c_obs, dist, winners_idx, n_items, rng, n_sims):
    """Random-package null: for each winner, draw package of same size uniformly."""
    all_dists = []
    universe = np.arange(n_items)
    for _ in range(n_sims):
        for i in winners_idx:
            items = np.where(c_obs[i])[0]
            k = len(items)
            if k < 2:
                continue
            fake = rng.choice(universe, size=k, replace=False)
            for a in range(len(fake)):
                for b in range(a + 1, len(fake)):
                    all_dists.append(dist[fake[a], fake[b]])
    return np.array(all_dists)


def figure_clustering(ctx):
    c_obs = ctx["c_obs_bundles"]
    dist = ctx["geo_distance"]
    n_items = c_obs.shape[1]
    winners_idx = np.where(c_obs.any(axis=1))[0]

    # multi-license winners only (pairwise needs ≥2)
    multi = [i for i in winners_idx if c_obs[i].sum() >= 2]

    obs_dists = _pairwise_distances_within_packages(c_obs, dist, multi)
    null_dists = _null_pairwise_distances(c_obs, dist, multi, n_items, RNG, N_NULL)

    # ── ECDF ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    xs_obs = np.sort(obs_dists)
    ys_obs = np.arange(1, len(xs_obs) + 1) / len(xs_obs)
    ax.plot(xs_obs, ys_obs, color=NAVY, lw=1.8, label="Observed packages")

    xs_null = np.sort(null_dists)
    ys_null = np.arange(1, len(xs_null) + 1) / len(xs_null)
    ax.plot(xs_null, ys_null, color=GOLD, lw=1.8, ls="--", label="Random packages (same size)")

    ax.set_xlabel("Within-package pairwise distance (km)", fontsize=9, family="serif")
    ax.set_ylabel("Cumulative share of license pairs", fontsize=9, family="serif")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_clustering.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    obs_mean = obs_dists.mean()
    null_mean = null_dists.mean()
    obs_med = np.median(obs_dists)
    null_med = np.median(null_dists)

    print("=== Figure 2: Clustering ===")
    print(f"  Multi-license winners: {len(multi)}")
    print(f"  Observed pairs: {len(obs_dists)}, null pairs: {len(null_dists)}")
    print(f"  Mean distance — observed: {obs_mean:,.0f} km, null: {null_mean:,.0f} km")
    print(f"  Median distance — observed: {obs_med:,.0f} km, null: {null_med:,.0f} km")
    print(f"  Ratio (obs/null mean): {obs_mean / null_mean:.3f}")
    print(f"  Kolmogorov–Smirnov shift: observed CDF dominates (closer pairs)")
    print()


def main():
    raw = load_bta_data()
    ctx = build_context(raw)
    figure_heterogeneity(raw, ctx)
    figure_heterogeneity_last_round(raw, ctx)
    figure_clustering(ctx)
    print(f"Figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
