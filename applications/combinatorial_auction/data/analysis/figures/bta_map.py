"""BTA map figures: winner packages and price heatmap."""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context, DATASETS_DIR
from applications.combinatorial_auction.data.analysis.style import NAVY, SLATE, DPI, style_ax

MAP_FILE = DATASETS_DIR / "bta_map_data" / "bta.mif"
OUT = Path(__file__).parent.parent / "output"

# Continental US bounding box
BBOX = (-125, 24, -66, 50)


def _load_geodata(raw):
    """Load BTA geometries and merge with auction data."""
    gdf = gpd.read_file(MAP_FILE)
    bta = raw["bta_data"]

    # Merge on BTA number
    gdf["bta"] = gdf["bta"].astype(int)
    gdf = gdf.merge(bta[["bta", "bidder_num_fox", "pop90", "bid"]],
                     on="bta", how="left")

    # Filter to continental US
    minx, miny, maxx, maxy = BBOX
    gdf = gdf.cx[minx:maxx, miny:maxy]
    return gdf


def plot_winners(raw, ctx, top_n=20):
    """Color top-N winners by package, rest in white."""
    gdf = _load_geodata(raw)
    c_obs = ctx["c_obs_bundles"]
    bidder = raw["bidder_data"]

    # Map bidder_num_fox to company name
    fox_to_name = dict(zip(bidder["bidder_num_fox"], bidder["co_name"]))

    # Count BTAs per winner
    winner_counts = gdf["bidder_num_fox"].value_counts()
    top_winners = winner_counts.head(top_n).index.tolist()

    # Assign colors
    cmap = colormaps["tab20"]
    color_map = {w: cmap(i / max(top_n - 1, 1)) for i, w in enumerate(top_winners)}

    gdf["color"] = gdf["bidder_num_fox"].map(
        lambda w: color_map.get(w, (0.95, 0.95, 0.95, 1.0))
    )

    fig, ax = plt.subplots(figsize=(14, 9))
    gdf.plot(color=gdf["color"].tolist(), edgecolor="black", linewidth=0.3, ax=ax)
    ax.axis("off")

    # Legend for top winners
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=color_map[w],
                       markersize=8, label=fox_to_name.get(w, str(w)))
               for w in top_winners]
    ax.legend(handles=handles, loc="lower left", fontsize=6, ncol=2,
              frameon=False, borderpad=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_bta_winners.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top {top_n} winners colored, {len(gdf)} continental BTAs plotted")


def plot_prices(raw):
    """Heatmap of winning bid prices across BTAs."""
    gdf = _load_geodata(raw)
    gdf["price_norm"] = gdf["bid"] / gdf["bid"].max()

    fig, ax = plt.subplots(figsize=(14, 9))
    gdf.plot(column="price_norm", ax=ax, legend=True, cmap="YlOrRd",
             edgecolor="black", linewidth=0.2, missing_kwds={"color": "white"},
             legend_kwds={"label": "Normalized price", "orientation": "horizontal",
                          "shrink": 0.5, "pad": 0.02})
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "fig_bta_prices.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Price map: max bid = ${gdf['bid'].max():,.0f}")


def plot_population(raw):
    """Heatmap of BTA population."""
    gdf = _load_geodata(raw)

    fig, ax = plt.subplots(figsize=(14, 9))
    gdf.plot(column="pop90", ax=ax, legend=True, cmap="Blues",
             edgecolor="black", linewidth=0.2, missing_kwds={"color": "white"},
             legend_kwds={"label": "Population (1990)", "orientation": "horizontal",
                          "shrink": 0.5, "pad": 0.02})
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "fig_bta_population.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Population map: max = {gdf['pop90'].max():,.0f}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot_winners(raw, ctx)
    plot_prices(raw)
    plot_population(raw)
