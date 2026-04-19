"""Jump-bid instrument columns for the 2SLS.

A bid at (market m, round r) is a JUMP if  bid_amt ≥ (1+κ)·min_accept,
where min_accept is the threshold published after round r−1 (bidders' "next
round floor"). Round-1 bids have no prior min_accept and are never jumps.

Exposes two column builders used by `iv._columns`:
    terminal_jumps(cutoff)   — per-BTA (count, $-excess) on jumps that were
                               the final bid on the license (no subsequent
                               bid submitted — the "overshoot" identification).
    jump_excess_after(cutoff) — per-BTA $-excess on ALL jumps after cutoff.
"""
import pandas as pd

from ...data.loaders import RAW, load_raw

JUMP_KAPPA = 0.025

_BID_FILE   = RAW / "cblock-submitted-bids.csv"
_MINAC_FILE = RAW / "cblock-minimum-accepted-bids-for-next-round.csv"


def _bta_code_to_int(series):
    """'B001' → 1, 'B321' → 321."""
    m = series.str.extract(r"^B(\d+)$", expand=False)
    if m.isna().any():
        raise ValueError(f"non-matching market codes: {series[m.isna()].unique()[:5]}")
    return m.astype(int)


def build_jump_table(kappa=JUMP_KAPPA):
    """Per-(bta, round) bid table with `is_jump` flag.

    The min_accept file is a concatenation of per-round snapshots: for
    file_round=r it re-states every (m, round≤r). We keep only each round's
    own snapshot (file_round == round_num) — the threshold bidders actually
    saw at the time. min_accept from round r applies to bids in round r+1.
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


def _align_to_continental(series, fill):
    """Reindex a bta-keyed Series to the continental BTA order used by the rest
    of the pipeline. Missing BTAs (no jumps in the window) → `fill`."""
    order = load_raw()["bta_data"]["bta"].astype(int).to_numpy()
    return series.reindex(order, fill_value=fill)


def terminal_jumps(round_cutoff=0, *, kappa=JUMP_KAPPA):
    """Per-BTA (count, excess_$B) of *terminal* jump bids after round_cutoff.

    A jump bid is "terminal" if no subsequent bid is submitted on the same
    license — the jumper's bid stands as the final action on j. The economic
    argument is that such a jump inflates price_j by an amount driven by the
    jumper's own strategic overshoot, not by any competitor's valuation.

    Returns (counts, excess_$B), both pandas Series indexed by continental bta.
    """
    m = build_jump_table(kappa)
    last_round = m.groupby("bta")["round_num"].max()
    m = m.join(last_round.rename("last_round"), on="bta")

    term = m[m["is_jump"] &
             (m["round_num"] == m["last_round"]) &
             (m["round_num"] > round_cutoff)].copy()
    term["excess_B"] = (term["bid_amt"] - term["min_accept"]) / 1e9

    return (_align_to_continental(term.groupby("bta").size(), fill=0),
            _align_to_continental(term.groupby("bta")["excess_B"].sum(), fill=0.0))


def jump_excess_after(round_cutoff, *, kappa=JUMP_KAPPA, normalize=True):
    """Per-BTA sum of (bid_amt − min_accept) across jumps in rounds > cutoff.

    Returns a pandas Series indexed by continental bta. If `normalize=True`,
    divides by Σ (winning bid) over all continental BTAs so the result is on
    the same share-of-revenue scale as ctx['price_share']; otherwise raw $.
    """
    m = build_jump_table(kappa)
    late = m[(m["round_num"] > round_cutoff) & m["is_jump"]].copy()
    late["excess"] = late["bid_amt"] - late["min_accept"]

    z = _align_to_continental(late.groupby("bta")["excess"].sum(), fill=0.0).astype(float)
    if normalize:
        z /= load_raw()["bta_data"]["bid"].astype(float).sum()
    return z
