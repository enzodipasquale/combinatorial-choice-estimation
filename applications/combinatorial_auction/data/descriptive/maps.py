"""Shared geographic data loading and heatmap plotting.

Use ``load_bta_gdf`` / ``load_mta_gdf`` to obtain continental GeoDataFrames
with auction data merged in, and ``plot_heatmap`` to render any column as a
choropleth.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

from applications.combinatorial_auction.data.loaders import DATASETS_DIR
from applications.combinatorial_auction.data.descriptive.style import DPI, style_ax

MAP_FILE = DATASETS_DIR / "bta_map_data" / "bta.mif"
XWALK_FILE = DATASETS_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data" / "cntysv2000_census-bta-may2009.csv"

# Continental US bounding box
BBOX = (-125, 24, -66, 50)


def _xwalk():
    df = pd.read_csv(XWALK_FILE, encoding="latin-1")[
        ["BTA", "MTA", "MTA Market Name"]
    ].drop_duplicates()
    df = df.rename(columns={"BTA": "bta", "MTA": "mta", "MTA Market Name": "mta_name"})
    df["bta"] = pd.to_numeric(df["bta"], errors="coerce")
    df["mta"] = pd.to_numeric(df["mta"], errors="coerce")
    df = df.dropna(subset=["bta", "mta"])
    df[["bta", "mta"]] = df[["bta", "mta"]].astype(int)
    return df


def load_bta_gdf(raw=None, extra=None):
    """Continental BTA GeoDataFrame.

    Parameters
    ----------
    raw : dict, optional
        Output of ``load_bta_data``; if given, merges ``bta``, ``bidder_num_fox``,
        ``pop90``, ``bid`` onto each polygon.
    extra : pd.DataFrame, optional
        Extra columns to merge on ``bta``.
    """
    gdf = gpd.read_file(MAP_FILE)
    gdf["bta"] = gdf["bta"].astype(int)
    if raw is not None:
        bta = raw["bta_data"][["bta", "bidder_num_fox", "pop90", "bid"]]
        gdf = gdf.merge(bta, on="bta", how="left")
    if extra is not None:
        gdf = gdf.merge(extra, on="bta", how="left")
    # Attach MTA id and name
    gdf = gdf.merge(_xwalk(), on="bta", how="left")
    minx, miny, maxx, maxy = BBOX
    return gdf.cx[minx:maxx, miny:maxy].copy()


def load_mta_gdf(extra=None):
    """Continental MTA GeoDataFrame (dissolved from BTAs).

    Parameters
    ----------
    extra : pd.DataFrame, optional
        Extra columns to merge on ``mta``.
    """
    gdf = gpd.read_file(MAP_FILE)
    gdf["bta"] = gdf["bta"].astype(int)
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
    """Choropleth of ``column`` over ``gdf``, saved to ``outfile``.

    Set ``log=True`` to color on log10 scale. ``label`` appears on the colorbar.
    """
    values = gdf[column].astype(float)
    if log:
        plot_vals = np.log10(values.where(values > 0))
        gdf = gdf.assign(_plot=plot_vals)
        col = "_plot"
        cbar_label = f"{label or column} (log$_{{10}}$)"
    else:
        col = column
        cbar_label = label or column

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
