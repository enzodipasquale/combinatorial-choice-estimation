"""Within-package distance ECDF: observed vs size-matched random null."""
import numpy as np
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import NAVY, GOLD, DPI, style_ax

N_NULL = 500


def _pair_dists(c_obs, dist, idx):
    out = []
    for i in idx:
        items = np.where(c_obs[i])[0]
        if len(items) < 2:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                out.append(dist[items[a], items[b]])
    return np.array(out)


def _null_dists(c_obs, dist, idx, rng, n_sims):
    universe = np.arange(c_obs.shape[1])
    out = []
    for _ in range(n_sims):
        for i in idx:
            k = c_obs[i].sum()
            if k < 2:
                continue
            fake = rng.choice(universe, size=k, replace=False)
            for a in range(len(fake)):
                for b in range(a + 1, len(fake)):
                    out.append(dist[fake[a], fake[b]])
    return np.array(out)


def plot(ctx):
    c_obs = ctx["c_obs_bundles"]
    dist = ctx["geo_distance"]
    multi = [i for i in np.where(c_obs.any(axis=1))[0] if c_obs[i].sum() >= 2]
    rng = np.random.default_rng(42)

    obs = _pair_dists(c_obs, dist, multi)
    null = _null_dists(c_obs, dist, multi, rng, N_NULL)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arr, color, ls, label in [
        (obs,  NAVY, "-",  "Observed packages"),
        (null, GOLD, "--", "Random packages (same size)"),
    ]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label)

    ax.set_xlabel("Within-package pairwise distance (km)", fontsize=9, family="serif")
    ax.set_ylabel("Cumulative share of license pairs", fontsize=9, family="serif")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_clustering.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  {len(multi)} multi-license winners; "
          f"mean obs {obs.mean():,.0f} km vs null {null.mean():,.0f} km "
          f"(ratio {obs.mean() / null.mean():.3f})")


if __name__ == "__main__":
    plot(build_context(load_raw()))
