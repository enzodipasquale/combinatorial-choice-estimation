"""Generic BTA/MTA heatmap pipeline.

    heatmap_bta(series_by_bta,  label=..., outfile=..., cmap=..., log=...)
    heatmap_mta(series_by_mta,  label=..., outfile=..., cmap=..., log=...)

Run as a script to regenerate the stock maps (pop, price, percapin).
"""
import numpy as np
import pandas as pd

from applications.combinatorial_auction.data.loaders import load_raw
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.maps import (
    load_bta_gdf, load_mta_gdf, plot_heatmap, xwalk,
)


def heatmap_bta(series, label, outfile, *, cmap="YlOrRd", log=False,
                vmin=None, vmax=None, units=None):
    """``series`` is a pd.Series indexed by BTA id."""
    extra = pd.DataFrame({"bta": series.index.astype(int), "_v": series.values})
    plot_heatmap(load_bta_gdf(extra=extra), "_v", outfile,
                 label=label, cmap=cmap, log=log,
                 vmin=vmin, vmax=vmax, units=units)


def heatmap_mta(series, label, outfile, *, cmap="YlOrRd", log=False,
                vmin=None, vmax=None, units=None):
    """``series`` is a pd.Series indexed by MTA id."""
    extra = pd.DataFrame({"mta": series.index.astype(int), "_v": series.values})
    plot_heatmap(load_mta_gdf(extra=extra), "_v", outfile,
                 label=label, cmap=cmap, log=log,
                 vmin=vmin, vmax=vmax, units=units)


def aggregate_to_mta(raw, column, weight=None):
    """Sum or weighted-mean of a BTA column up to MTA level. Returns pd.Series."""
    bta = raw["bta_data"].merge(xwalk(), on="bta", how="left").dropna(subset=["mta"])
    if weight is None:
        return bta.groupby("mta")[column].sum()
    return bta.groupby("mta").apply(
        lambda d: np.average(d[column], weights=d[weight]) if d[weight].sum() > 0 else np.nan
    )


if __name__ == "__main__":
    raw = load_raw()
    bta = raw["bta_data"]
    bta_ids = bta["bta"].astype(int).values

    heatmap_bta(pd.Series(bta["pop90"].values, index=bta_ids),
                label="Population (1990)",
                outfile=OUT_FIG / "fig_bta_population.png", cmap="Blues", log=True)
    heatmap_bta(pd.Series(bta["bid"].values, index=bta_ids),
                label="Winning bid",
                outfile=OUT_FIG / "fig_bta_prices.png", cmap="YlOrRd",
                log=True, units="USD")
    heatmap_bta(pd.Series(bta["percapin"].values, index=bta_ids),
                label="Per-capita income",
                outfile=OUT_FIG / "fig_bta_percapin.png", cmap="Greens",
                units="USD")

    heatmap_mta(aggregate_to_mta(raw, "pop90"),
                label="MTA population (1990)",
                outfile=OUT_FIG / "fig_mta_population.png", cmap="Blues", log=True)
    heatmap_mta(aggregate_to_mta(raw, "bid"),
                label="MTA total winning bids",
                outfile=OUT_FIG / "fig_mta_prices.png", cmap="YlOrRd",
                log=True, units="USD")
