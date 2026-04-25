"""Jump-bid descriptive: round-by-round share of bids that are jumps.

Definition (FCC C-block).
    A bid in round r on market m is a *jump* if
        bid_amt ≥ (1 + κ) · min_accept_{m,r},        κ = JUMP_KAPPA = 2.5%
    where min_accept is the threshold published after round r-1.  Round-1
    bids have no prior min_accept and are never jumps.

Run
    python -m applications.combinatorial_auction.data.descriptive.figures.jump_bids
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from applications.combinatorial_auction.data.loaders import RAW
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, GOLD, SLATE, DPI, style_ax,
)


JUMP_KAPPA = 0.025

_BID_FILE   = RAW / "cblock-submitted-bids.csv"
_MINAC_FILE = RAW / "cblock-minimum-accepted-bids-for-next-round.csv"


# ── jump-bid table ────────────────────────────────────────────────────

def _bta_code_to_int(series: pd.Series) -> pd.Series:
    """'B001' → 1, 'B321' → 321."""
    m = series.str.extract(r"^B(\d+)$", expand=False)
    if m.isna().any():
        raise ValueError(f"non-matching market codes: {series[m.isna()].unique()[:5]}")
    return m.astype(int)


def build_jump_table(kappa: float = JUMP_KAPPA) -> pd.DataFrame:
    """Per-(bta, round) bid table with an `is_jump` flag.

    The min_accept file is a concatenation of per-round snapshots: for
    file_round = r it re-states every (m, round ≤ r).  We keep only each
    round's own snapshot (`file_round == round_num`) — the threshold bidders
    saw at the time.  The min_accept from round r applies to bids in round
    r + 1 (so we shift `round_num` by +1 before merging).
    """
    bids  = pd.read_csv(_BID_FILE)
    minac = pd.read_csv(_MINAC_FILE)

    bids = bids.assign(bta=_bta_code_to_int(bids["market"]))
    minac = minac[minac["file_round"] == minac["round_num"]]
    minac = (minac[["market", "round_num", "min_accept"]]
                  .rename(columns={"round_num": "prev_round"}))
    minac["round_num"] = minac["prev_round"] + 1

    m = bids.merge(minac[["market", "round_num", "min_accept"]],
                   on=["market", "round_num"], how="left")
    m["is_jump"] = (m["min_accept"].notna() &
                    (m["bid_amt"] >= (1 + kappa) * m["min_accept"]))
    return m


# ── figure ────────────────────────────────────────────────────────────

def plot(kappa: float = JUMP_KAPPA, outfile: Path | None = None) -> None:
    if outfile is None:
        outfile = OUT_FIG / "fig_jump_bids.png"

    m = build_jump_table(kappa)
    by_r = (m.groupby("round_num")
                 .agg(n_bids=("bid_amt", "size"), n_jumps=("is_jump", "sum"))
                 .astype({"n_jumps": int}))
    share = by_r["n_jumps"] / by_r["n_bids"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()

    # Left axis — counts.
    l_all = ax.plot(by_r.index, by_r["n_bids"].values, color=SLATE, lw=1.3,
                    marker="o", markersize=2.8, markerfacecolor=SLATE,
                    markeredgecolor="white", markeredgewidth=0.3,
                    label="All bids", zorder=2)[0]
    l_jmp = ax.plot(by_r.index, by_r["n_jumps"].values, color=NAVY, lw=1.8,
                    marker="o", markersize=3.2, markerfacecolor=NAVY,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label="Jump bids", zorder=3)[0]

    # Right axis — share of jumps.
    l_sh = ax2.plot(by_r.index, share.values, color=GOLD, lw=1.4, ls="--",
                    label="Share of jumps", zorder=4)[0]

    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["right"].set_color(GOLD)

    ax.set_xlabel("Auction round", fontsize=10, family="serif")
    ax.set_ylabel("Number of bids", fontsize=10, family="serif")
    ax2.set_ylabel("Share of bids that are jumps", fontsize=10,
                   family="serif", color=GOLD)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(0, 1)
    ax.set_xlim(by_r.index.min() - 0.5, by_r.index.max() + 0.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax2.tick_params(axis="y", colors=GOLD, labelsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6, zorder=0)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()
                  + ax2.get_yticklabels()):
        label.set_family("serif")
    ax.legend(handles=[l_all, l_jmp, l_sh], fontsize=9, frameon=False,
              loc="upper right", prop={"family": "serif", "size": 9})

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    n_bids  = int(by_r["n_bids"].sum())
    n_jumps = int(by_r["n_jumps"].sum())

    def window(mask, label):
        s = by_r.loc[mask, ["n_jumps", "n_bids"]].sum()
        sh   = s["n_jumps"] / s["n_bids"] if s["n_bids"] else float("nan")
        conc = s["n_jumps"] / n_jumps if n_jumps else float("nan")
        print(f"    {label:<14}  {int(s['n_jumps']):>4} jumps / {int(s['n_bids']):>5} bids  "
              f"= {sh:6.1%}  ({conc:5.1%} of all jumps)")

    print(f"  fig_jump_bids: κ={kappa:.1%}, rounds {by_r.index.min()}..{by_r.index.max()}")
    print(f"    total bids   = {n_bids:,}")
    print(f"    total jumps  = {n_jumps:,}  ({n_jumps / n_bids:.1%} of all bids)")
    print()
    print(f"    {'window':<14}  {'jumps / bids':>21}  {'share':>9}  {'concentration':>17}")
    window(by_r.index <= 10, "rounds 1-10")
    window(by_r.index <= 50, "rounds ≤ 50")
    window(by_r.index >  50, "rounds > 50")
    print()
    cumj = by_r["n_jumps"].cumsum()
    for q in (0.50, 0.75, 0.90, 0.95):
        r = int((cumj >= q * n_jumps).idxmax())
        print(f"    {q:.0%} of all jumps occurred by round {r}")


if __name__ == "__main__":
    plot()
