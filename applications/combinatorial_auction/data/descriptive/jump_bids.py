"""Descriptive inspection of the round-level bid data.

Utility functions for eyeballing the jump-bid distribution; not on the
estimation path. The actual 2SLS instrument columns live in
`scripts/second_stage/jump_bids.py`.

Usage:
    python -m applications.combinatorial_auction.data.descriptive.jump_bids [plot]
"""
import sys
from pathlib import Path
import pandas as pd

from ...scripts.second_stage.jump_bids import (
    JUMP_KAPPA, build_jump_table, _bta_code_to_int, _BID_FILE, _MINAC_FILE,
)
from ..loaders import RAW, load_raw

PLOT_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "jump_bids_timeseries.png"


def inspect():
    """One-shot sanity pass over the raw round files."""
    bids  = pd.read_csv(_BID_FILE)
    minac = pd.read_csv(_MINAC_FILE)
    raw   = load_raw()
    cont  = set(raw["bta_data"]["bta"].astype(int).tolist())

    print(f"submitted_bids      shape={bids.shape}   rounds {bids['round_num'].min()}..{bids['round_num'].max()}")
    print(f"min_accepted_next   shape={minac.shape}")

    for name, df in [("submitted_bids", bids), ("min_accepted_next", minac)]:
        ints = _bta_code_to_int(pd.Series(df["market"].unique()))
        print(f"  {name:<22} unique markets={len(ints)}   continental overlap={len(set(ints) & cont)}/{len(cont)}")

    bpr = bids.groupby("round_num").size()
    print(f"  bids/round  mean={bpr.mean():.1f}  median={bpr.median():.0f}  max={bpr.max()}  last-10={bpr.iloc[-10:].sum()}")

    b_known = set(raw["bidder_data"]["bidder_num"].unique())
    b_bids  = set(bids["bidder_num"].unique())
    print(f"  bidder_num overlap: {len(b_bids & b_known)}/{len(b_bids)}; in-bids-only={sorted(b_bids - b_known)}")


def plot_timeseries(kappa=JUMP_KAPPA, out_path=PLOT_PATH):
    """Overlay total submitted bids and jump-bid counts per round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = build_jump_table(kappa)
    by_r = m.groupby("round_num").agg(n_bids=("bid_amt", "size"), n_jumps=("is_jump", "sum"))

    print(f"κ={kappa:.1%}  rounds {by_r.index.min()}..{by_r.index.max()}  "
          f"Σbids={by_r['n_bids'].sum()}  Σjumps={int(by_r['n_jumps'].sum())} "
          f"({100*by_r['n_jumps'].sum()/by_r['n_bids'].sum():.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_r.index, by_r["n_bids"],  label="submitted bids", color="steelblue", lw=1.5)
    ax.plot(by_r.index, by_r["n_jumps"], label=f"jump bids (≥ {1+kappa:.3f}×min_accept)",
            color="crimson", lw=1.5)
    ax.set_xlabel("round"); ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"C-block: submitted vs jump bids per round  (κ={kappa:.1%})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    (plot_timeseries if (len(sys.argv) > 1 and sys.argv[1] == "plot") else inspect)()
