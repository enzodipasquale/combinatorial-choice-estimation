"""750 km geographic balls around the NY and LA BTA centroids.

Motivates the gravity-complementarity specification by showing, on the same
reference map used in the counterfactual (continental US with MTA boundaries),
how far a coast-to-centroid radius reaches.
"""
import matplotlib.pyplot as plt
import geopandas as gpd

from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import NAVY, GOLD, SLATE, DPI
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf, load_mta_gdf

NY_BTA, LA_BTA = 321, 262
R_KM = 750


def _ball(bta_m, bta_id, radius_km):
    row = bta_m[bta_m["bta"] == bta_id].iloc[0]
    return row.geometry.centroid.buffer(radius_km * 1000)


def plot(bta_id, label, color, outfile, radius_km=R_KM):
    bta = load_bta_gdf()
    mta = load_mta_gdf()
    bta_m = bta.to_crs(5070)  # USA Contiguous Albers Equal-Area (meters)

    ball = gpd.GeoSeries([_ball(bta_m, bta_id, radius_km)],
                         crs=5070).to_crs(bta.crs)
    c = bta[bta["bta"] == bta_id].iloc[0].geometry.centroid

    fig, ax = plt.subplots(figsize=(12, 6))
    bta.plot(ax=ax, facecolor="white", edgecolor="#BDBDBD", linewidth=0.15)
    mta.plot(ax=ax, facecolor="none", edgecolor=NAVY, linewidth=1.1)
    ball.plot(ax=ax, facecolor=color, alpha=0.28,
              edgecolor=color, linewidth=1.6)
    ax.plot(c.x, c.y, marker="o", ms=5, color=color, zorder=5)

    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.axis("off")
    ax.text(0.02, 0.05,
            f"Ball of radius {radius_km} km around {label}",
            transform=ax.transAxes, fontsize=10, family="serif", color=SLATE)
    fig.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  {outfile.name}  (centroid lon={c.x:.2f}, lat={c.y:.2f})")


if __name__ == "__main__":
    plot(NY_BTA, "NY", NAVY, OUT_FIG / f"fig_ball_ny_{R_KM}km.png")
    plot(LA_BTA, "LA", GOLD, OUT_FIG / f"fig_ball_la_{R_KM}km.png")
