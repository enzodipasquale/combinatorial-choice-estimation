"""BTA maps: top-20 winners by eligibility, price and population heatmaps."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import DPI
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf, plot_heatmap


def plot_winners(raw, ctx, top_n=20):
    """Top-N winners by eligibility in distinct colors, others in gray."""
    gdf = load_bta_gdf(raw)
    bidder = raw["bidder_data"]
    elig = bidder["pops_eligible"].values

    win_idx = np.where(ctx["c_obs_bundles"].any(axis=1))[0]
    top_idx = win_idx[np.argsort(elig[win_idx])[::-1][:top_n]]
    top_fox = bidder.iloc[top_idx]["bidder_num_fox"].tolist()
    fox_to_name = dict(zip(bidder["bidder_num_fox"], bidder["co_name"]))

    cmap = colormaps["tab20"]
    color_map = {fox: cmap(i / max(top_n - 1, 1)) for i, fox in enumerate(top_fox)}
    OTHER, UNWON = (0.82, 0.82, 0.84, 1.0), (1.0, 1.0, 1.0, 1.0)
    gdf["color"] = gdf["bidder_num_fox"].map(
        lambda f: UNWON if pd.isna(f) else color_map.get(int(f), OTHER)
    )

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.01, 0.02, 0.72, 0.96])
    legend_ax = fig.add_axes([0.74, 0.02, 0.25, 0.96]); legend_ax.axis("off")
    gdf.plot(color=gdf["color"].tolist(), edgecolor="black", linewidth=0.3, ax=ax)
    ax.axis("off"); ax.set_aspect("equal")

    handles = []
    for fox in top_fox:
        i = list(bidder["bidder_num_fox"]).index(fox)
        lab = f"{fox_to_name.get(fox, str(fox))[:32]}  ({elig[i] / 1e6:.1f}M)"
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=color_map[fox], markersize=10, label=lab))
    handles.append(Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=OTHER, markersize=10, label="Other winners"))
    legend_ax.legend(handles=handles, loc="center left", fontsize=8, frameon=False,
                     borderpad=0.3, labelspacing=0.6,
                     title=f"Top {top_n} winners by eligibility", title_fontsize=9)

    fig.savefig(OUT_FIG / "fig_bta_winners.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_bta_winners: top {top_n} by eligibility")


def plot_prices(raw):
    gdf = load_bta_gdf(raw).assign(bid_m=lambda d: d["bid"] / 1e6)
    plot_heatmap(gdf, "bid_m", OUT_FIG / "fig_bta_prices.png",
                 label="Winning bid (USD millions)", cmap="YlOrRd", log=True)


def plot_population(raw):
    gdf = load_bta_gdf(raw).assign(pop_m=lambda d: d["pop90"] / 1e6)
    plot_heatmap(gdf, "pop_m", OUT_FIG / "fig_bta_population.png",
                 label="Population 1990 (millions)", cmap="Blues", log=True)


if __name__ == "__main__":
    raw = load_raw()
    ctx = build_context(raw)
    plot_winners(raw, ctx)
    plot_prices(raw)
    plot_population(raw)
