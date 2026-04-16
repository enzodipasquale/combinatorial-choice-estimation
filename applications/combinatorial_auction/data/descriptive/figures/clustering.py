"""Within-package distance ECDF: observed vs random null."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, GOLD, DPI, style_ax,
)
from applications.combinatorial_auction.data.descriptive.helpers import (
    pairwise_distances_within_packages, null_pairwise_distances,
)

OUT = Path(__file__).parent.parent / "output"
N_NULL = 500


def plot(ctx):
    c_obs = ctx["c_obs_bundles"]
    dist = ctx["geo_distance"]
    winners_idx = np.where(c_obs.any(axis=1))[0]
    multi = [i for i in winners_idx if c_obs[i].sum() >= 2]

    rng = np.random.default_rng(42)
    obs_dists = pairwise_distances_within_packages(c_obs, dist, multi)
    null_dists = null_pairwise_distances(c_obs, dist, multi, rng, N_NULL)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    xs_obs = np.sort(obs_dists)
    ys_obs = np.arange(1, len(xs_obs) + 1) / len(xs_obs)
    ax.plot(xs_obs, ys_obs, color=NAVY, lw=1.8, label="Observed packages")

    xs_null = np.sort(null_dists)
    ys_null = np.arange(1, len(xs_null) + 1) / len(xs_null)
    ax.plot(xs_null, ys_null, color=GOLD, lw=1.8, ls="--",
            label="Random packages (same size)")

    ax.set_xlabel("Within-package pairwise distance (km)", fontsize=9, family="serif")
    ax.set_ylabel("Cumulative share of license pairs", fontsize=9, family="serif")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT / "fig_clustering.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  Multi-license winners: {len(multi)}")
    print(f"  Mean distance — observed: {obs_dists.mean():,.0f} km, null: {null_dists.mean():,.0f} km")
    print(f"  Median distance — observed: {np.median(obs_dists):,.0f} km, null: {np.median(null_dists):,.0f} km")
    print(f"  Ratio (obs/null mean): {obs_dists.mean() / null_dists.mean():.3f}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot(ctx)
