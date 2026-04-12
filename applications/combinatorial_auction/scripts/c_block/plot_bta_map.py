#!/usr/bin/env python3
"""BTA map colored by winner, ordered by eligibility (top N colored, rest white)."""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

# ── Load data ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
raw = load_bta_data()
ctx = build_context(raw)
bta_data = raw["bta_data"]
bidder_data = raw["bidder_data"]

b_obs = ctx["c_obs_bundles"]  # (n_obs, n_btas)
bta_nums = bta_data["bta"].astype(int).values
bidder_names = bidder_data["co_name"].values
elig = bidder_data["pops_eligible"].astype(float).values

# ── Map BTA -> winner (bidder index) ───────────────────────────
n_obs, n_btas = b_obs.shape
bta_winner = {}  # bta_num -> bidder_index
for j in range(n_btas):
    col = b_obs[:, j]
    if col.any():
        bta_winner[bta_nums[j]] = col.argmax()

# ── Rank bidders by eligibility (descending) ───────────────────
TOP_N = 20  # number of bidders to color
elig_rank = np.argsort(elig)[::-1]
top_bidders = set(elig_rank[:TOP_N])

# Assign winner labels
bta_labels = {}
for bta_num, bidder_idx in bta_winner.items():
    if bidder_idx in top_bidders:
        bta_labels[bta_num] = bidder_names[bidder_idx]
    else:
        bta_labels[bta_num] = "_other"

# ── Load shapefile ─────────────────────────────────────────────
gdf = gpd.read_file(DATA_DIR / "bta_shapefile" / "bta.shp")
gdf["BTA"] = gdf["BTA"].astype(int)
gdf["winner_label"] = gdf["BTA"].map(bta_labels).fillna("_none")

# Continental US bounding box
minx, miny, maxx, maxy = -125, 24, -66, 50
cont = gdf.cx[minx:maxx, miny:maxy].copy()

# ── Color map: top bidders get distinct colors, rest white ─────
# Order legend by eligibility (descending)
top_names_ordered = []
for idx in elig_rank[:TOP_N]:
    name = bidder_names[idx]
    if name in cont["winner_label"].values:
        top_names_ordered.append(name)

cmap = colormaps["tab20"]
color_map = {}
for i, name in enumerate(top_names_ordered):
    color_map[name] = cmap(i / max(len(top_names_ordered) - 1, 1))
color_map["_other"] = (0.85, 0.85, 0.85, 1.0)  # light gray
color_map["_none"] = (1.0, 1.0, 1.0, 1.0)       # white (not in our data)

cont["color"] = cont["winner_label"].map(color_map)

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 16), dpi=200)
cont.plot(color=cont["color"].tolist(), edgecolor="black", linewidth=0.3, ax=ax)

# Legend: top bidders by eligibility
legend_handles = []
for name in top_names_ordered:
    short = name[:35]
    idx = list(bidder_names).index(name)
    label = f"{short}  ({elig[idx]/1e6:.1f}M)"
    legend_handles.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_map[name],
               markersize=12, label=label))
legend_handles.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor=color_map["_other"],
           markersize=12, label="Other winners"))

ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
          fontsize=10, frameon=True, title="Winners (by eligibility)", title_fontsize=12)

ax.axis("off")
ax.set_title("C-Block BTA Winners by Bidder Eligibility", fontsize=22, fontweight='bold', pad=20)

outpath = Path(__file__).parent / "bta_winners_map.png"
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {outpath}")
