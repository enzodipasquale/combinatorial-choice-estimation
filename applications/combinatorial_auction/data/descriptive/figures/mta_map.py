"""MTA maps: boundaries with labels, and MTA-vs-BTA contrast."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf, load_mta_gdf


def plot_boundaries():
    mta = load_mta_gdf()
    fig, ax = plt.subplots(figsize=(14, 8.5))
    mta.plot(ax=ax, facecolor="white", edgecolor=NAVY, linewidth=1.1)
    for _, row in mta.iterrows():
        if pd.isna(row["mta_name"]):
            continue
        pt = row.geometry.representative_point()
        ax.annotate(row["mta_name"].strip().split(",")[0][:22],
                    xy=(pt.x, pt.y), fontsize=5.5, ha="center", va="center",
                    family="serif", color=SLATE)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_mta_boundaries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_mta_boundaries: {len(mta)} MTAs")


def plot_vs_bta():
    """MTAs white-filled with navy outlines, BTAs as hairlines on top."""
    bta = load_bta_gdf()
    mta = load_mta_gdf().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8.5))
    mta.plot(ax=ax, facecolor="white",
             edgecolor=NAVY, linewidth=1.4, zorder=1)
    bta.plot(ax=ax, facecolor="none", edgecolor="black",
             linewidth=0.25, alpha=0.55, zorder=2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_mta_vs_bta.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_mta_vs_bta: {len(mta)} MTAs over {len(bta)} BTAs")


if __name__ == "__main__":
    plot_boundaries()
    plot_vs_bta()
