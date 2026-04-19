"""Figure for the JMP slides: jump-bid share by round in the C-block auction.

A bid is a *jump* if ``bid_amt ≥ (1+κ)·min_accept`` (κ = JUMP_KAPPA = 2.5%),
per the definition in ``jump_bids.build_jump_table``.  The figure shows the
share of bids that are jumps against round number, with a faint bar chart of
bid volume on a secondary axis so the audience can read the denominator.
A dashed marker sits at the default ``jump_cutoff`` (round 50) — after that
point jump bidding is essentially extinct and the remaining rounds are noisy
because of tiny bid counts.

Run:
    python -m applications.combinatorial_auction.scripts.second_stage.plot_jump_bids
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .jump_bids import JUMP_KAPPA, build_jump_table

NAVY  = "#1B2A4A"
GOLD  = "#B8860B"
SLATE = "#4A6274"
VOL   = "#D6DBE2"   # faint slate/blue for bid-volume bars

_REPO  = Path(__file__).resolve().parents[4]
OUTFIG = _REPO / "slides" / "artifacts" / "figures" / "fig_jump_bids.png"

JUMP_CUTOFF_DEFAULT = 50   # matches scripts/second_stage/iv.py default


def plot(kappa=JUMP_KAPPA, cutoff=JUMP_CUTOFF_DEFAULT, outfile=OUTFIG):
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

    # Right axis — share of jump bids.
    l_sh = ax2.plot(by_r.index, share.values, color=GOLD, lw=1.4, ls="--",
                    label="Share of jumps", zorder=4)[0]

    for side in ("top",):
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["right"].set_color(GOLD)

    ax.set_xlabel("Auction round", fontsize=10, family="serif")
    ax.set_ylabel("Number of bids", fontsize=10, family="serif")
    ax2.set_ylabel("Share of bids that are jumps", fontsize=10, family="serif",
                   color=GOLD)
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
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary stats — aggregate so we're not comparing tiny tail rounds.
    n_bids  = int(by_r["n_bids"].sum())
    n_jumps = int(by_r["n_jumps"].sum())
    e = by_r.loc[by_r.index <= 10, ["n_jumps", "n_bids"]].sum()
    pre = by_r.loc[by_r.index <= cutoff, ["n_jumps", "n_bids"]].sum()
    post = by_r.loc[by_r.index > cutoff, ["n_jumps", "n_bids"]].sum()

    # Round R* at which cumulative future jump share stays below 5%.
    cum_future_bids  = by_r["n_bids"].iloc[::-1].cumsum().iloc[::-1]
    cum_future_jumps = by_r["n_jumps"].iloc[::-1].cumsum().iloc[::-1]
    tail_share = cum_future_jumps / cum_future_bids.replace(0, np.nan)
    r_five = int(tail_share[tail_share <= 0.05].index.min()) if (tail_share <= 0.05).any() else None

    print(f"  fig_jump_bids: κ={kappa:.1%}, rounds {by_r.index.min()}..{by_r.index.max()}")
    print(f"    total bids  = {n_bids:,}")
    print(f"    total jumps = {n_jumps:,} ({n_jumps / n_bids:.1%})")
    print(f"    rounds 1–10      share = {e['n_jumps']/e['n_bids']:.1%}  "
          f"({int(e['n_jumps'])}/{int(e['n_bids'])})")
    print(f"    rounds ≤ {cutoff}       share = {pre['n_jumps']/pre['n_bids']:.1%}  "
          f"({int(pre['n_jumps'])}/{int(pre['n_bids'])})")
    print(f"    rounds > {cutoff}       share = {post['n_jumps']/post['n_bids']:.1%}  "
          f"({int(post['n_jumps'])}/{int(post['n_bids'])})")
    print(f"    cumulative tail share ≤ 5% from round {r_five} onward")


if __name__ == "__main__":
    plot()
