"""MTA map: dissolve BTA geometries into Major Trading Areas."""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, DATASETS_DIR
from applications.combinatorial_auction.data.descriptive.style import NAVY, SLATE, DPI, style_ax

MAP_FILE = DATASETS_DIR / "bta_map_data" / "bta.mif"
RAW_DIR = DATASETS_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"
OUT = Path(__file__).parent.parent / "output"

BBOX = (-125, 24, -66, 50)


def _build_mta_geodata():
    """Dissolve BTA polygons into MTAs using the census crosswalk."""
    gdf = gpd.read_file(MAP_FILE)
    gdf["bta"] = gdf["bta"].astype(int)

    xwalk = pd.read_csv(RAW_DIR / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    xwalk = xwalk[["BTA", "MTA", "MTA Market Name"]].drop_duplicates()
    xwalk = xwalk.rename(columns={"BTA": "bta", "MTA": "mta", "MTA Market Name": "mta_name"})
    xwalk["bta"] = pd.to_numeric(xwalk["bta"], errors="coerce")
    xwalk["mta"] = pd.to_numeric(xwalk["mta"], errors="coerce")
    xwalk = xwalk.dropna(subset=["bta", "mta"])
    xwalk["bta"] = xwalk["bta"].astype(int)
    xwalk["mta"] = xwalk["mta"].astype(int)

    gdf = gdf.merge(xwalk[["bta", "mta", "mta_name"]], on="bta", how="left")
    gdf = gdf.dropna(subset=["mta"])

    # Dissolve BTAs into MTAs
    mta_gdf = gdf.dissolve(by="mta", as_index=False)

    # Keep MTA name (take first)
    names = xwalk.drop_duplicates("mta").set_index("mta")["mta_name"]
    mta_gdf["mta_name"] = mta_gdf["mta"].map(names)

    return mta_gdf


def plot_mta_boundaries():
    """Plain MTA boundary map with labels."""
    mta_gdf = _build_mta_geodata()

    # Continental filter
    minx, miny, maxx, maxy = BBOX
    cont = mta_gdf.cx[minx:maxx, miny:maxy].copy()

    fig, ax = plt.subplots(figsize=(14, 9))
    cont.plot(ax=ax, facecolor="white", edgecolor=NAVY, linewidth=1.2)

    # Label each MTA at its centroid
    for _, row in cont.iterrows():
        pt = row.geometry.representative_point()
        name = row["mta_name"]
        if pd.isna(name):
            continue
        # Shorten long names
        short = name.strip()[:20]
        ax.annotate(short, xy=(pt.x, pt.y), fontsize=4, ha="center", va="center",
                    family="serif", color=SLATE)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mta_boundaries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  MTA boundary map: {len(cont)} continental MTAs")


def plot_mta_bta_overlay():
    """MTA boundaries (thick) overlaid on BTA boundaries (thin)."""
    mta_gdf = _build_mta_geodata()
    bta_gdf = gpd.read_file(MAP_FILE)

    minx, miny, maxx, maxy = BBOX
    cont_mta = mta_gdf.cx[minx:maxx, miny:maxy]
    cont_bta = bta_gdf.cx[minx:maxx, miny:maxy]

    fig, ax = plt.subplots(figsize=(14, 9))
    cont_bta.plot(ax=ax, facecolor="white", edgecolor=SLATE, linewidth=0.2, alpha=0.5)
    cont_mta.plot(ax=ax, facecolor="none", edgecolor=NAVY, linewidth=1.5)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mta_bta_overlay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  MTA/BTA overlay: {len(cont_mta)} MTAs, {len(cont_bta)} BTAs")


if __name__ == "__main__":
    plot_mta_boundaries()
    plot_mta_bta_overlay()
