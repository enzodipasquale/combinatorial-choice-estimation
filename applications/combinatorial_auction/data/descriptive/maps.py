"""BTA/MTA GeoDataFrame loaders and a generic choropleth plot function."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import DATASETS, RAW
from applications.combinatorial_auction.data.descriptive.style import DPI

MAP_FILE = DATASETS / "bta_map_data" / "bta.mif"
XWALK_FILE = RAW / "cntysv2000_census-bta-may2009.csv"
BBOX = (-125, 24, -66, 50)  # continental US


def _xwalk():
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
    gdf = gdf.merge(_xwalk(), on="bta", how="left")
    minx, miny, maxx, maxy = BBOX
    return gdf.cx[minx:maxx, miny:maxy].copy()


def load_mta_gdf(extra=None):
    """Continental MTA GeoDataFrame (dissolved from BTAs)."""
    gdf = gpd.read_file(MAP_FILE).assign(bta=lambda d: d["bta"].astype(int))
    gdf = gdf.merge(_xwalk(), on="bta", how="left").dropna(subset=["mta"])
    mta = gdf.dissolve(by="mta", as_index=False)
    names = _xwalk().drop_duplicates("mta").set_index("mta")["mta_name"]
    mta["mta_name"] = mta["mta"].map(names)
    if extra is not None:
        mta = mta.merge(extra, on="mta", how="left")
    minx, miny, maxx, maxy = BBOX
    return mta.cx[minx:maxx, miny:maxy].copy()


def plot_heatmap(gdf, column, outfile, *, label=None, cmap="YlOrRd",
                 log=False, edgecolor="white", linewidth=0.25,
                 figsize=(11, 7), vmin=None, vmax=None):
    """Choropleth of ``gdf[column]`` saved to ``outfile``.

    ``log=True`` colors on log10 scale (zeros become missing).
    """
    values = gdf[column].astype(float)
    if log:
        gdf = gdf.assign(_plot=np.log10(values.where(values > 0)))
        col, cbar_label = "_plot", f"{label or column} (log$_{{10}}$)"
    else:
        col, cbar_label = column, label or column

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.84])
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.025])
    gdf.plot(column=col, ax=ax, legend=True, cmap=cmap,
             edgecolor=edgecolor, linewidth=linewidth,
             missing_kwds={"color": "whitesmoke", "edgecolor": edgecolor,
                           "linewidth": linewidth},
             vmin=vmin, vmax=vmax, cax=cax,
             legend_kwds={"label": cbar_label, "orientation": "horizontal"})
    ax.axis("off")
    ax.set_aspect("equal")
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
