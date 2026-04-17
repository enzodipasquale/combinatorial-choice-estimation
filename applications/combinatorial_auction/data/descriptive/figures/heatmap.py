"""Heatmap pipeline for arbitrary BTA- or MTA-level variables.

Examples
--------
    from applications.combinatorial_auction.data.descriptive.figures.heatmap import heatmap_bta, heatmap_mta

    # pass a pd.Series indexed by bta number (continental is enforced)
    heatmap_bta(series, label="Per-capita income (USD)", outfile="fig_bta_income.png")

    # or a raw numpy vector aligned with load_bta_data()['bta_data']['bta']
    heatmap_bta(values, label="...", outfile="...", by="bta_order", raw=raw)

Run as a script to generate the stock heatmaps used in the slides
(population, price, per-capita income) at MTA and BTA level.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data
from applications.combinatorial_auction.data.descriptive.maps import (
    load_bta_gdf, load_mta_gdf, plot_heatmap,
)

OUT = Path(__file__).parent.parent / "output"


def _series_from(values, raw, level):
    """Accept a Series (indexed by bta/mta id) or raw ndarray in bta_data order."""
    if isinstance(values, pd.Series):
        return values
    if level == "bta":
        ids = raw["bta_data"]["bta"].astype(int).values
    else:
        raise ValueError("For MTA-level ndarray input, pass a pd.Series indexed by mta id.")
    return pd.Series(np.asarray(values, dtype=float), index=ids)


def heatmap_bta(values, label, outfile, *, raw=None, cmap="YlOrRd", log=False,
                vmin=None, vmax=None):
    """Render a BTA-level heatmap. ``values`` is a pd.Series indexed by BTA id
    or a 1-D array in the order of ``raw['bta_data']``.
    """
    if raw is None and not isinstance(values, pd.Series):
        raw = load_bta_data()
    s = _series_from(values, raw, "bta")
    gdf = load_bta_gdf(raw, extra=pd.DataFrame({"bta": s.index, "_v": s.values}))
    plot_heatmap(gdf, "_v", Path(outfile), label=label, cmap=cmap,
                 log=log, vmin=vmin, vmax=vmax)


def heatmap_mta(values, label, outfile, *, cmap="YlOrRd", log=False,
                vmin=None, vmax=None):
    """Render an MTA-level heatmap. ``values`` is a pd.Series indexed by MTA id."""
    if not isinstance(values, pd.Series):
        raise ValueError("heatmap_mta requires a pd.Series indexed by MTA id.")
    extra = pd.DataFrame({"mta": values.index.astype(int), "_v": values.values})
    gdf = load_mta_gdf(extra=extra)
    plot_heatmap(gdf, "_v", Path(outfile), label=label, cmap=cmap,
                 log=log, vmin=vmin, vmax=vmax)


def _aggregate_bta_to_mta(raw, column, weight=None):
    """Aggregate a BTA column up to MTA level.

    If ``weight`` is given, computes weighted mean; otherwise sum.
    """
    from applications.combinatorial_auction.data.descriptive.maps import _xwalk
    bta = raw["bta_data"].merge(_xwalk(), on="bta", how="left").dropna(subset=["mta"])
    if weight is None:
        return bta.groupby("mta")[column].sum()
    w = bta[weight]
    return bta.groupby("mta").apply(
        lambda d: np.average(d[column], weights=d[weight]) if d[weight].sum() > 0 else np.nan
    )


if __name__ == "__main__":
    raw = load_bta_data()
    bta = raw["bta_data"]

    # ---- BTA-level heatmaps ----
    heatmap_bta(
        pd.Series(bta["pop90"].values / 1e6, index=bta["bta"].astype(int).values),
        label="Population 1990 (millions)",
        outfile=OUT / "fig_bta_population.png", cmap="Blues", log=True,
    )
    heatmap_bta(
        pd.Series(bta["bid"].values / 1e6, index=bta["bta"].astype(int).values),
        label="Winning bid (USD millions)",
        outfile=OUT / "fig_bta_prices.png", cmap="YlOrRd", log=True,
    )
    heatmap_bta(
        pd.Series(bta["percapin"].values, index=bta["bta"].astype(int).values),
        label="Per-capita income (USD)",
        outfile=OUT / "fig_bta_percapin.png", cmap="Greens",
    )

    # ---- MTA-level heatmaps (aggregate up) ----
    pop_mta = _aggregate_bta_to_mta(raw, "pop90") / 1e6
    heatmap_mta(pop_mta, label="MTA population 1990 (millions)",
                outfile=OUT / "fig_mta_population.png", cmap="Blues")

    bid_mta = _aggregate_bta_to_mta(raw, "bid") / 1e6
    heatmap_mta(bid_mta, label="MTA total winning bids (USD millions)",
                outfile=OUT / "fig_mta_prices.png", cmap="YlOrRd", log=True)

    print(f"  BTA heatmaps: population, prices (log), percapin")
    print(f"  MTA heatmaps: population, prices (log)")
