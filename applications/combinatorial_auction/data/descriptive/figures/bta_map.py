"""BTA maps: top winners by eligibility / package size, price and population heatmaps.

Conference-ready design: 12 unique (shade, hatch) combinations on a clean white
map.  Two shades of blue (dark = top 6, medium = ranks 7-12), six distinct
hatches per shade, for 12 combos with no two bidders sharing the same key.
The legend lives in its own axes column on the right of the figure so it
never overlaps the map.

For the package-size ranking we add a 13th category in gold solid fill:
"Single-license winners" — every bidder who won exactly one BTA.  The
``_mta`` variants additionally overlay MTA boundaries in a darker, slightly
thicker line on top of the BTA grid.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import DPI
from applications.combinatorial_auction.data.descriptive.maps import load_bta_gdf

# Dark navy for the top 6, medium blue for ranks 7-12.
SHADE_DARK = "#33507C"   # softened navy — still dark but not inky
SHADE_MID  = "#7BA1C9"   # lighter mid-blue to keep the band contrast
SHADE_SOLO = "#B8860B"   # gold accent — single-license winners stand apart

# Six visually distinct hatch patterns per shade band.
HATCHES_TOP = ["",        "////",   "xxxx",   "....",   "||||",  "++++"]
HATCHES_MID = ["",        "\\\\\\\\", "oooo", "----",   "****",  "OOOO"]

BG_EDGE   = "#6F6F6F"   # gray borders for every BTA
MAIN_EDGE = "#6F6F6F"   # same gray for colored bidders
MTA_EDGE  = "#000000"   # near-black for the MTA overlay
HATCH_LW  = 0.55


def _key(rank):
    """(shade, hatch) for a 0-based rank in 0..11."""
    if rank < 6:
        return SHADE_DARK, HATCHES_TOP[rank]
    return SHADE_MID, HATCHES_MID[rank - 6]


def _short_label(co_name, value, unit):
    """'NextWave Personal Communications Inc.' → 'NextWave (176M)'."""
    name = co_name.replace(",", "").strip()
    for suffix in ("Inc.", "Inc", "L.P.", "LP", "L.L.C.", "LLC", "Corp.", "Corp"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    toks = name.split()
    if toks and toks[0].upper() == "THE":
        toks = toks[1:]
    head = toks[0] if toks else name
    if len(toks) > 1 and toks[0].lower() in {"cook", "pcs"}:
        head = f"{toks[0]} {toks[1]}"
    elif len(toks) > 1 and toks[1].isupper() and len(toks[1]) <= 4:
        head = f"{toks[0]} {toks[1]}"
    if unit == "M":
        return f"{head} ({value / 1e6:.0f}M)"
    return f"{head} ({int(value)} BTAs)"


def _plot_top_winners(raw, ctx, *, metric, outfile, top_n=12,
                      show_singles=False):
    gdf = load_bta_gdf(raw)
    bidder = raw["bidder_data"]
    c_obs = ctx["c_obs_bundles"]
    elig = bidder["pops_eligible"].values
    pkg_size = c_obs.sum(axis=1)

    win_mask = c_obs.any(axis=1)
    win_idx = np.where(win_mask)[0]
    if metric == "elig":
        key, unit = elig, "M"
    elif metric == "pkg_size":
        key, unit = pkg_size, "BTAs"
    else:
        raise ValueError(metric)

    order = win_idx[np.argsort(key[win_idx])[::-1]]
    top_fox = bidder.iloc[order[:top_n]]["bidder_num_fox"].astype(int).tolist()
    fox_to_name = dict(zip(bidder["bidder_num_fox"].astype(int), bidder["co_name"]))
    fox_to_key  = dict(zip(bidder["bidder_num_fox"].astype(int), key))

    # Single-license winners: not in the top-N, pkg_size == 1.
    if show_singles:
        singles_mask = win_mask & (pkg_size == 1)
        singles_fox = set(bidder.loc[singles_mask, "bidder_num_fox"].astype(int).tolist())
        singles_fox -= set(top_fox)
        n_singles = int(singles_mask.sum() - len([f for f in top_fox
                                                  if fox_to_key[f] == 1 and unit == "BTAs"]))
    else:
        singles_fox, n_singles = set(), 0

    # Hatch styling is global — set before any polygon is drawn.
    plt.rcParams["hatch.linewidth"] = HATCH_LW
    plt.rcParams["hatch.color"]     = "#FFFFFF"

    # Height picked so the map fills its cell with no vertical whitespace:
    # map axes ≈ 9.1" wide, map aspect ≈ 59°/(26°/1.25) ≈ 2.84 → 5.2" tall.
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = GridSpec(1, 2, width_ratios=[2.7, 1], wspace=0.02,
                  left=0.005, right=0.995, top=0.99, bottom=0.01)
    ax  = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1]); lax.axis("off")

    # 1. Clean white US with thin gray BTA borders.
    gdf.plot(ax=ax, facecolor="white", edgecolor=BG_EDGE, linewidth=0.25)

    # 2. Top-N bidders: shade + white hatch + gray contour (two-pass).
    for rank, fox in enumerate(top_fox):
        sub = gdf[gdf["bidder_num_fox"].fillna(-1).astype(int) == fox]
        if sub.empty:
            continue
        shade, hatch = _key(rank)
        sub.plot(ax=ax, facecolor=shade, edgecolor="#FFFFFF",
                 linewidth=0, hatch=hatch)
        sub.plot(ax=ax, facecolor="none", edgecolor=MAIN_EDGE, linewidth=0.25)

    # 3. Optional 13th category: single-license winners (gold solid).
    if show_singles and singles_fox:
        sub = gdf[gdf["bidder_num_fox"].fillna(-1).astype(int).isin(singles_fox)]
        if not sub.empty:
            sub.plot(ax=ax, facecolor=SHADE_SOLO, edgecolor=MAIN_EDGE, linewidth=0.25)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.axis("off")

    # Legend — top-N always, gold singles slot optional, no "Other / non-winners" row.
    handles, labels = [], []
    for rank, fox in enumerate(top_fox):
        shade, hatch = _key(rank)
        handles.append((
            Patch(facecolor=shade, edgecolor="#FFFFFF", linewidth=0, hatch=hatch),
            Patch(facecolor="none", edgecolor=MAIN_EDGE, linewidth=0.6),
        ))
        labels.append(_short_label(fox_to_name[fox], fox_to_key[fox], unit))
    if show_singles and singles_fox:
        handles.append((
            Patch(facecolor=SHADE_SOLO, edgecolor="#FFFFFF", linewidth=0),
            Patch(facecolor="none", edgecolor=MAIN_EDGE, linewidth=0.6),
        ))
        labels.append(f"Single-license winners ({len(singles_fox)})")

    header = "Top 12 by eligibility" if metric == "elig" else "Top 12 by package size"
    lax.legend(
        handles=handles, labels=labels,
        loc="center left", bbox_to_anchor=(0.0, 0.5),
        fontsize=9.5, frameon=False, borderpad=0.2, labelspacing=0.85,
        handletextpad=0.9, handleheight=1.6, handlelength=2.3,
        prop={"family": "serif", "size": 9.5},
        title=header, title_fontsize=10, alignment="left",
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    tag = "eligibility" if metric == "elig" else "package size"
    print(f"  {outfile.name}: top {top_n} winners by {tag}")
    for rank, fox in enumerate(top_fox):
        name = fox_to_name[fox][:36]
        nb = int((gdf["bidder_num_fox"] == fox).sum())
        v = fox_to_key[fox]
        v_str = f"{v/1e6:6.1f}M elig" if unit == "M" else f"{int(v):3d} BTAs"
        shade, hatch = _key(rank)
        print(f"    {name:36}  {nb:3d} BTAs  ({v_str})  "
              f"[{shade} {hatch or '(solid)'}]")
    if show_singles and singles_fox:
        print(f"    single-license winners: {len(singles_fox)} bidders [gold]")


def plot_winners_by_pkg_size(raw, ctx, top_n=12):
    _plot_top_winners(raw, ctx, metric="pkg_size",
                      outfile=OUT_FIG / "fig_bta_winners_pkg.png",
                      top_n=top_n, show_singles=True)


if __name__ == "__main__":
    plot_winners_by_pkg_size(load_raw(), build_context(load_raw()))
