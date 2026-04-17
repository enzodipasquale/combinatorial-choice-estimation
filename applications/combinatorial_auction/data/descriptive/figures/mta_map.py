"""MTA-level map figures: boundaries, BTA overlay, and MTA vs BTA contrast."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path

from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf, load_mta_gdf

OUT = Path(__file__).parent.parent / "output"


def plot_boundaries():
    """Plain MTA boundary map with labels at centroids."""
    mta = load_mta_gdf()

    fig, ax = plt.subplots(figsize=(14, 8.5))
    mta.plot(ax=ax, facecolor="white", edgecolor=NAVY, linewidth=1.1)

    for _, row in mta.iterrows():
        if pd.isna(row["mta_name"]):
            continue
        pt = row.geometry.representative_point()
        short = row["mta_name"].strip().split(",")[0][:22]
        ax.annotate(short, xy=(pt.x, pt.y), fontsize=5.5, ha="center", va="center",
                    family="serif", color=SLATE)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mta_boundaries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_mta_boundaries: {len(mta)} continental MTAs")


def plot_vs_bta():
    """High-contrast MTA/BTA comparison: MTAs colored, BTAs hairline inside.

    Communicates at a glance: a coarse MTA market contains many fine BTAs.
    """
    bta = load_bta_gdf()
    mta = load_mta_gdf()

    # Distinct but muted fill colors for MTAs
    cmap = colormaps["tab20"]
    n = len(mta)
    mta = mta.reset_index(drop=True)
    mta["_color"] = [cmap(i % 20) for i in range(n)]
    # Lighten
    mta["_fill"] = mta["_color"].apply(lambda c: (c[0], c[1], c[2], 0.30))

    fig, ax = plt.subplots(figsize=(14, 8.5))
    mta.plot(ax=ax, color=mta["_fill"].tolist(), edgecolor=NAVY, linewidth=1.4,
             zorder=1)
    # BTA subdivision lines on top
    bta.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.25,
             alpha=0.55, zorder=2)

    ax.axis("off")

    # Corner caption in the figure itself (small, unobtrusive)
    ax.text(0.02, 0.03,
            f"{len(mta)} MTAs (colored)  ·  {len(bta)} BTAs (thin lines)",
            transform=ax.transAxes, fontsize=9, family="serif", color=SLATE)

    fig.tight_layout()
    fig.savefig(OUT / "fig_mta_vs_bta.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_mta_vs_bta: {len(mta)} MTAs over {len(bta)} BTAs")


if __name__ == "__main__":
    plot_boundaries()
    plot_vs_bta()
