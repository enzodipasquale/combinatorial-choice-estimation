"""BTA/MTA GeoDataFrame loaders and a generic choropleth plot function."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import FuncFormatter

from applications.combinatorial_auction.data.loaders import DATASETS, RAW
from applications.combinatorial_auction.data.descriptive.style import DPI

MAP_FILE = DATASETS / "bta_map_data" / "bta.mif"
XWALK_FILE = RAW / "cntysv2000_census-bta-may2009.csv"
BBOX = (-125, 24, -66, 50)  # continental US


def xwalk():
    df = pd.read_csv(XWALK_FILE, encoding="latin-1")[
        ["BTA", "MTA", "MTA Market Name"]
    ].drop_duplicates().rename(
        columns={"BTA": "bta", "MTA": "mta", "MTA Market Name": "mta_name"}
    )
    df["bta"] = pd.to_numeric(df["bta"], errors="coerce")
    df["mta"] = pd.to_numeric(df["mta"], errors="coerce")
    return df.dropna(subset=["bta", "mta"]).astype({"bta": int, "mta": int})


def load_bta_gdf(raw=None, extra=None):
    """Continental BTA GeoDataFrame, optionally merged with auction data."""
    gdf = gpd.read_file(MAP_FILE)
    gdf["bta"] = gdf["bta"].astype(int)
    if raw is not None:
        gdf = gdf.merge(raw["bta_data"][["bta", "bidder_num_fox", "pop90", "bid"]],
                         on="bta", how="left")
    if extra is not None:
        gdf = gdf.merge(extra, on="bta", how="left")
    gdf = gdf.merge(xwalk(), on="bta", how="left")
    minx, miny, maxx, maxy = BBOX
    return gdf.cx[minx:maxx, miny:maxy].copy()


def load_mta_gdf(extra=None):
    """Continental MTA GeoDataFrame (dissolved from BTAs)."""
    gdf = gpd.read_file(MAP_FILE).assign(bta=lambda d: d["bta"].astype(int))
    gdf = gdf.merge(xwalk(), on="bta", how="left").dropna(subset=["mta"])
    mta = gdf.dissolve(by="mta", as_index=False)
    names = xwalk().drop_duplicates("mta").set_index("mta")["mta_name"]
    mta["mta_name"] = mta["mta"].map(names)
    if extra is not None:
        mta = mta.merge(extra, on="mta", how="left")
    minx, miny, maxx, maxy = BBOX
    return mta.cx[minx:maxx, miny:maxy].copy()


def _short(x):
    """Short human label: 30K, 1.2M, $3.5B."""
    a = abs(x)
    if a >= 1e9:  return f"{x/1e9:g}B"
    if a >= 1e6:  return f"{x/1e6:g}M"
    if a >= 1e3:  return f"{x/1e3:g}K"
    if a == 0:    return "0"
    return f"{x:g}"


def plot_heatmap(gdf, column, outfile, *, label=None, cmap="YlOrRd",
                 log=False, edgecolor="white", linewidth=0.25,
                 figsize=(11, 6.5), vmin=None, vmax=None, units=None):
    """Choropleth of ``gdf[column]`` saved to ``outfile``.

    ``log=True``  → log-norm colorbar, ticks printed as real values (K/M/B).
    ``units``     → 'USD' prepends a dollar sign on tick labels.
    """
    values = gdf[column].astype(float)
    finite = values[np.isfinite(values)]
    if log:
        pos = finite[finite > 0]
        vmin = pos.min() if vmin is None else vmin
        vmax = pos.max() if vmax is None else vmax
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = finite.min() if vmin is None else vmin
        vmax = finite.max() if vmax is None else vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(column=column, ax=ax, cmap=cmap, norm=norm,
             edgecolor=edgecolor, linewidth=linewidth,
             missing_kwds={"color": "whitesmoke", "edgecolor": edgecolor,
                           "linewidth": linewidth})
    ax.axis("off")
    ax.set_aspect("equal")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    prefix = "$" if units == "USD" else ""
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        shrink=0.55, aspect=35, pad=0.02, fraction=0.045)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: prefix + _short(x)))
    cbar.ax.tick_params(labelsize=9)
    if label:
        cbar.set_label(label, fontsize=10, family="serif")

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
