"""BTA map figures: winner packages (top-N by eligibility) and heatmaps."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI, style_ax
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf, plot_heatmap

OUT = Path(__file__).parent.parent / "output"


def plot_winners(raw, ctx, top_n=20):
    """Color the top-N bidders (by eligibility) that won licenses.

    Remaining winners shown in light gray; unwon BTAs in white. Ranking by
    eligibility highlights the assortative-matching story from the slides.
    """
    gdf = load_bta_gdf(raw)
    bidder = raw["bidder_data"]
    elig = bidder["pops_eligible"].values

    # Rank bidders by eligibility (descending) restricted to those that won
    won = ctx["c_obs_bundles"].any(axis=1)
    won_idx = np.where(won)[0]
    top_idx = won_idx[np.argsort(elig[won_idx])[::-1][:top_n]]

    fox_to_name = dict(zip(bidder["bidder_num_fox"], bidder["co_name"]))
    top_fox = bidder.iloc[top_idx]["bidder_num_fox"].tolist()

    cmap = colormaps["tab20"]
    color_map = {fox: cmap(i / max(top_n - 1, 1)) for i, fox in enumerate(top_fox)}
    OTHER = (0.82, 0.82, 0.84, 1.0)
    UNWON = (1.0, 1.0, 1.0, 1.0)

    def _color(fox):
        if pd.isna(fox):
            return UNWON
        return color_map.get(int(fox), OTHER)

    gdf["color"] = gdf["bidder_num_fox"].map(_color)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.01, 0.02, 0.72, 0.96])
    legend_ax = fig.add_axes([0.74, 0.02, 0.25, 0.96])
    legend_ax.axis("off")

    gdf.plot(color=gdf["color"].tolist(), edgecolor="black",
             linewidth=0.3, ax=ax)
    ax.axis("off")
    ax.set_aspect("equal")

    handles = []
    for fox in top_fox:
        name = fox_to_name.get(fox, str(fox))
        i = list(bidder["bidder_num_fox"]).index(fox)
        lab = f"{name[:32]}  ({elig[i] / 1e6:.1f}M)"
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=color_map[fox],
                              markersize=10, label=lab))
    handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=OTHER,
                          markersize=10, label="Other winners"))

    legend_ax.legend(handles=handles, loc="center left",
                     fontsize=8, frameon=False, borderpad=0.3,
                     labelspacing=0.6,
                     title=f"Top {top_n} winners by eligibility",
                     title_fontsize=9)

    fig.savefig(OUT / "fig_bta_winners.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_bta_winners: top {top_n} by eligibility, {len(gdf)} continental BTAs")


def plot_prices(raw):
    gdf = load_bta_gdf(raw)
    gdf["bid_m"] = gdf["bid"] / 1e6
    plot_heatmap(gdf, "bid_m", OUT / "fig_bta_prices.png",
                 label="Winning bid (USD millions)", cmap="YlOrRd", log=True)
    print(f"  fig_bta_prices: max bid = ${gdf['bid'].max():,.0f}")


def plot_population(raw):
    gdf = load_bta_gdf(raw)
    gdf["pop_m"] = gdf["pop90"] / 1e6
    plot_heatmap(gdf, "pop_m", OUT / "fig_bta_population.png",
                 label="Population 1990 (millions)", cmap="Blues", log=True)
    print(f"  fig_bta_population: max pop = {gdf['pop90'].max():,.0f}")


if __name__ == "__main__":
    raw = load_bta_data()
    ctx = build_context(raw)
    plot_winners(raw, ctx)
    plot_prices(raw)
    plot_population(raw)
