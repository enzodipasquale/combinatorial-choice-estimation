"""Round-level bid data loader & sanity inspection.

Exploratory — does not touch the estimation pipeline. Goal: load the FCC
C-block round-by-round bid files, join them, reconcile the `market` codes
(B001, B002, …) with the `bta` integers used by the rest of the project,
and eyeball the distributions so we can build a jump-bid instrument.

Usage:
    python -m applications.combinatorial_auction.scripts.second_stage.jump_bids
    python -m applications.combinatorial_auction.scripts.second_stage.jump_bids plot
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from ...data.loaders import RAW, load_raw

JUMP_KAPPA = 0.025   # a bid is a "jump" if bid_amt >= (1 + κ) · min_accept_for_round
PLOT_PATH  = Path(__file__).resolve().parent.parent.parent / "results" / "jump_bids_timeseries.png"

BID_FILE   = RAW / "cblock-submitted-bids.csv"
HBW_FILE   = RAW / "cblock-high-bids-before-withdrawal.csv"
MINAC_FILE = RAW / "cblock-minimum-accepted-bids-for-next-round.csv"


def load_round_data():
    """Read the three round-level CSVs. No joins yet."""
    bids  = pd.read_csv(BID_FILE)
    hbw   = pd.read_csv(HBW_FILE)
    minac = pd.read_csv(MINAC_FILE)
    return bids, hbw, minac


def bta_code_to_int(series):
    """'B001' → 1, 'B321' → 321. Raises if any label doesn't match /^B\\d+$/."""
    m = series.str.extract(r"^B(\d+)$", expand=False)
    if m.isna().any():
        bad = series[m.isna()].unique()
        raise ValueError(f"non-matching market codes: {bad[:5]}")
    return m.astype(int)


def inspect():
    bids, hbw, minac = load_round_data()
    raw = load_raw()
    cont_btas = set(raw["bta_data"]["bta"].astype(int).tolist())

    print("=" * 72)
    print("RAW FILE SHAPES")
    print("=" * 72)
    for name, df in [("submitted_bids", bids), ("high_bids_before_wd", hbw),
                     ("min_accepted_next", minac)]:
        print(f"  {name:<22} {df.shape}   cols={df.columns.tolist()}")
    print()

    # ── Markets present ────────────────────────────────────────────────
    print("=" * 72)
    print("MARKET CODES")
    print("=" * 72)
    for name, df in [("submitted_bids", bids), ("min_accepted_next", minac)]:
        markets = df["market"].unique()
        as_int  = bta_code_to_int(pd.Series(markets))
        overlap = set(as_int.tolist()) & cont_btas
        only_f  = set(as_int.tolist()) - cont_btas
        only_c  = cont_btas - set(as_int.tolist())
        print(f"  {name:<22} {len(markets)} unique markets;"
              f" continental overlap={len(overlap)}/{len(cont_btas)};"
              f" file-only={len(only_f)}; continental-only={len(only_c)}")
    print()

    # ── Round structure ────────────────────────────────────────────────
    print("=" * 72)
    print("ROUND STRUCTURE")
    print("=" * 72)
    for name, df in [("submitted_bids", bids), ("min_accepted_next", minac)]:
        r = df["round_num"]
        print(f"  {name:<22} rounds {r.min()}..{r.max()}   n_distinct={r.nunique()}")
    # Quick view of how many submitted bids per round
    bids_per_round = bids.groupby("round_num").size()
    print(f"  bids/round   mean={bids_per_round.mean():.1f}   "
          f"median={bids_per_round.median():.0f}   "
          f"max={bids_per_round.max()}   "
          f"last-10-rounds total={bids_per_round.iloc[-10:].sum()}")
    print()

    # ── Bidder IDs ─────────────────────────────────────────────────────
    print("=" * 72)
    print("BIDDER IDs (submitted_bids.bidder_num vs biddercblk)")
    print("=" * 72)
    bidder_map = raw["bidder_data"][["bidder_num", "bidder_num_fox"]]
    sub_bidders = set(bids["bidder_num"].unique())
    known       = set(bidder_map["bidder_num"].unique())
    print(f"  unique bidder_num in submitted_bids: {len(sub_bidders)}")
    print(f"  unique bidder_num in bidder_data:   {len(known)}")
    print(f"  overlap:                            {len(sub_bidders & known)}")
    print(f"  in-bids-only:                       {len(sub_bidders - known)}"
          f" (e.g. {sorted(sub_bidders - known)[:10]})")
    print()

    # ── bid_amt magnitudes ─────────────────────────────────────────────
    print("=" * 72)
    print("BID AMOUNTS (raw, submitted_bids.bid_amt)")
    print("=" * 72)
    x = bids["bid_amt"].astype(float)
    print(f"  count={len(x)}   min={x.min():.0f}   median={x.median():.0f}   "
          f"mean={x.mean():.0f}   max={x.max():.3e}")
    # Sanity-check against final bid prices in btadata (billions / 1e9).
    final_bid_by_bta = raw["bta_data"].set_index("bta")["bid"].astype(float)
    print(f"  btadata 'bid' column (winning price):   min={final_bid_by_bta.min():.0f}"
          f"   median={final_bid_by_bta.median():.0f}   max={final_bid_by_bta.max():.3e}")
    print("  (both should be in the same $ units — verify below)")
    print()

    # ── Does the last-round submitted bid per license = the btadata bid? ─
    print("=" * 72)
    print("CROSS-CHECK: final-round submitted bid  vs  btadata.bid")
    print("=" * 72)
    bids = bids.copy()
    bids["bta"] = bta_code_to_int(bids["market"])
    last = (bids.sort_values(["bta", "round_num"])
                 .groupby("bta")
                 .tail(1)[["bta", "round_num", "bid_amt", "bidder_num"]]
                 .set_index("bta"))
    join = last.join(final_bid_by_bta.rename("btadata_bid"), how="inner")
    join["diff"] = join["bid_amt"].astype(float) - join["btadata_bid"].astype(float)
    print(f"  n continental BTAs with a submitted-bid history: {len(join)}")
    print(f"  |diff| == 0:          {(join['diff'].abs() == 0).sum()}")
    print(f"  |diff| <= 1:          {(join['diff'].abs() <= 1).sum()}")
    print(f"  |diff|/btadata <=1e-3:{((join['diff'].abs() / join['btadata_bid']) <= 1e-3).sum()}")
    print(f"  worst abs diff:       {join['diff'].abs().max():.0f}")
    print(join.sort_values("diff", key=lambda s: s.abs(), ascending=False).head(5).to_string())


def build_jump_table(kappa=JUMP_KAPPA):
    """Per-(market, round) table with bid counts and jump-bid counts.

    A bid at (market m, round r) is a JUMP if  bid_amt ≥ (1 + κ) · min_accept,
    where min_accept is the threshold published after round r−1 (i.e. the file
    row for (m, r−1)). Round 1 has no prior min_accept → not classifiable,
    those bids are counted as non-jump.
    """
    bids, _hbw, minac = load_round_data()
    bids = bids.assign(bta=bta_code_to_int(bids["market"]))

    # The min_accept file is a concatenation of per-round snapshots: for
    # file_round = r, the file re-states (previous_h, min_accept) for every
    # round ≤ r. Keep only each round's own snapshot (file_round == round_num)
    # — that's the threshold bidders actually saw at the time.
    minac = minac[minac["file_round"] == minac["round_num"]]
    minac = minac.rename(columns={"round_num": "prev_round"})[["market", "prev_round", "min_accept"]]
    minac["round_num"] = minac["prev_round"] + 1     # min_accept applies to the NEXT round

    m = bids.merge(minac[["market", "round_num", "min_accept"]],
                   on=["market", "round_num"], how="left")
    m["is_jump"] = (m["min_accept"].notna() &
                    (m["bid_amt"] >= (1 + kappa) * m["min_accept"]))
    return m


def jump_bids_after(round_cutoff, *, kappa=JUMP_KAPPA):
    """Per-BTA count of jump bids submitted in rounds > round_cutoff.

    Returns a pandas Series indexed by `bta` (continental ints) aligned with
    raw["bta_data"]["bta"]. Missing BTAs (no jumps in the window) get 0.
    """
    m = build_jump_table(kappa)
    late = m[(m["round_num"] > round_cutoff) & m["is_jump"]]
    counts = late.groupby("bta").size()

    raw = load_raw()
    order = raw["bta_data"]["bta"].astype(int).to_numpy()
    return counts.reindex(order, fill_value=0)


def terminal_jumps(round_cutoff=0, *, kappa=JUMP_KAPPA):
    """Per-BTA (count, excess_$B) of *terminal* jump bids after round_cutoff.

    A jump bid is "terminal" if no subsequent bid is submitted on the same
    license — i.e. the jumper's bid stands as the final action on j. The
    argument is that such a jump inflates price_j by an amount driven by the
    jumper's own strategic overshoot, not by any competitor's valuation.

    Returns two pandas Series indexed by continental `bta`:
        count  = # terminal jump bids on j in rounds > cutoff
        excess = Σ (bid_amt − min_accept) across those bids, in $B
    """
    m = build_jump_table(kappa)

    # For each license, the round of its very last bid (any bidder, any type).
    last_round = m.groupby("bta")["round_num"].max()
    m = m.join(last_round.rename("last_round"), on="bta")

    term = m[m["is_jump"] &
             (m["round_num"] == m["last_round"]) &
             (m["round_num"] > round_cutoff)].copy()
    term["excess_B"] = (term["bid_amt"] - term["min_accept"]) / 1e9

    counts  = term.groupby("bta").size()
    excess  = term.groupby("bta")["excess_B"].sum()

    raw   = load_raw()
    order = raw["bta_data"]["bta"].astype(int).to_numpy()
    return (counts.reindex(order, fill_value=0),
            excess.reindex(order, fill_value=0.0))


def jump_excess_after(round_cutoff, *, kappa=JUMP_KAPPA, normalize=True):
    """Per-BTA sum of jump-bid *excess over min_accept* in rounds > round_cutoff.

    excess_j = Σ_{jumps in j, round > R*} (bid_amt − min_accept)

    If normalize=True (default), divide by Σ_k (winning price_k) over all
    continental BTAs so the resulting variable is on the same share-of-total-
    revenue scale as `bid_share` (ctx["price_share"]).
    """
    m = build_jump_table(kappa)
    late = m[(m["round_num"] > round_cutoff) & m["is_jump"]].copy()
    late["excess"] = late["bid_amt"] - late["min_accept"]
    sums = late.groupby("bta")["excess"].sum()

    raw = load_raw()
    order = raw["bta_data"]["bta"].astype(int).to_numpy()
    z = sums.reindex(order, fill_value=0.0).astype(float)
    if normalize:
        z = z / raw["bta_data"]["bid"].astype(float).sum()
    return z


def inspect_instrument(kappa=JUMP_KAPPA, cutoffs=(0, 10, 25, 50, 75, 100)):
    """Print summary statistics of the jump-bid count for several cutoffs."""
    print(f"JUMP-BID INSTRUMENT — κ = {kappa:.3%}")
    print(f"{'cutoff R*':>10} {'Σ jumps':>10} {'#BTAs>0':>10} "
          f"{'mean':>8} {'median':>8} {'max':>6}")
    for c in cutoffs:
        z = jump_bids_after(c, kappa=kappa)
        print(f"{c:>10d} {int(z.sum()):>10d} {int((z > 0).sum()):>10d} "
              f"{z.mean():>8.2f} {int(z.median()):>8d} {int(z.max()):>6d}")


def plot_timeseries(kappa=JUMP_KAPPA, out_path=PLOT_PATH):
    """Overlay: total submitted bids per round and jump-bid count per round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = build_jump_table(kappa)
    by_round = m.groupby("round_num").agg(n_bids=("bid_amt", "size"),
                                          n_jumps=("is_jump", "sum"))
    by_round["share"] = by_round["n_jumps"] / by_round["n_bids"].replace(0, np.nan)

    print(f"κ = {kappa:.3%}")
    print(f"  rounds: {by_round.index.min()}..{by_round.index.max()}")
    print(f"  Σ bids  = {by_round['n_bids'].sum()}")
    print(f"  Σ jumps = {int(by_round['n_jumps'].sum())}  "
          f"({100*by_round['n_jumps'].sum()/by_round['n_bids'].sum():.1f}%)")
    print(f"  round-1 bids (not classifiable, counted as non-jump): "
          f"{int(by_round.loc[1, 'n_bids'])}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_round.index, by_round["n_bids"],  label="all submitted bids",
            color="steelblue", lw=1.5)
    ax.plot(by_round.index, by_round["n_jumps"], label=f"jump bids (≥ {1+kappa:.3f}×min_accept)",
            color="crimson", lw=1.5)
    ax.set_xlabel("round")
    ax.set_ylabel("count")
    ax.set_title(f"C-block: submitted vs jump bids per round  (κ = {kappa:.1%})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if cmd == "plot":
        plot_timeseries()
    elif cmd == "instr":
        inspect_instrument()
    else:
        inspect()
