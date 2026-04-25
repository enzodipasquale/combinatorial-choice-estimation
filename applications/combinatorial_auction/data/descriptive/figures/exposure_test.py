"""Reduced-form test of the exposure problem in the C-block auction.

For each bidder i who LOSES A FLAGSHIP late in the auction (as defined in
the writeup), we look at i's behavior on the complementary licenses they
were also SHB on at the moment of the loss, in the next 30 rounds:

  withdrew   — i withdrew that complementary license
  re-engaged — i bid on it, or on a license within 200 km of it
  stuck      — neither, AND i bid on *something* in the auction during the
               look-ahead window  (active everywhere except on this fragment)
  exited     — neither, and i was completely silent (uninformative for the
               exposure question; reported separately)

Single canonical spec:  late-phase = last 20% of rounds, distance ≤ 500 km
for "complementary", look-ahead = 30 rounds, ≥ 3 consecutive SHB rounds.
Robustness deferred unless results are sensitive (we report the main spec
and one robustness on the geographic-window radius).

Run:
    python -m applications.combinatorial_auction.data.descriptive.figures.exposure_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import load_raw, RAW
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, GOLD, DPI, style_ax,
)

# Reuse the data-loading helpers from the prior resilience script.
from applications.combinatorial_auction.data.descriptive.figures.resilience import (
    _load_shb, _load_bids, _load_withdrawals, _load_distance_km,
    _CONSECUTIVE_MIN,
)

# ── canonical parameters (committed up front) ────────────────────────
LATE_FRAC      = 0.20    # last 20% of rounds
COMPL_KM       = 500     # complementary = within this many km of flagship
NEAR_KM        = 200     # "re-engaged" if bid within this many km of j'
LOOK_AHEAD     = 30      # rounds after loss to observe behavior


def _classify(bidder, j_prime, t, *, look_ahead, near_km, idx_of, d_km,
              bids_by_it, withdraw):
    """Return one of: 'withdrew' | 're-engaged' | 'stuck' | 'exited'."""
    end = t + look_ahead

    # withdrawal of j' by this bidder?
    for r in range(t + 1, end + 1):
        if (r, j_prime, bidder) in withdraw:
            return "withdrew"

    # nearby licenses to j'
    if j_prime in idx_of:
        ji = idx_of[j_prime]
        near = {bid for bid, k in idx_of.items()
                if k != ji and d_km[ji, k] <= near_km}
        near.add(j_prime)
    else:
        near = {j_prime}

    bid_on_near = False
    bid_on_anything = False
    for r in range(t + 1, end + 1):
        bids_r = bids_by_it.get((bidder, r), set())
        if not bids_r:
            continue
        bid_on_anything = True
        if bids_r & near:
            bid_on_near = True
            break

    if bid_on_near:
        return "re-engaged"
    if bid_on_anything:
        return "stuck"
    return "exited"


def run(*, late_frac=LATE_FRAC, compl_km=COMPL_KM, near_km=NEAR_KM,
        look_ahead=LOOK_AHEAD, tag="main"):
    raw = load_raw()
    bta = raw["bta_data"]
    bta_ids = bta["bta"].astype(int).to_numpy()
    pops    = bta["pop90"].astype(float).to_numpy()
    d_km    = _load_distance_km(bta_ids)
    idx_of  = {int(b): k for k, b in enumerate(bta_ids)}

    shb_wide = _load_shb()
    bids_df  = _load_bids()
    withdraw = _load_withdrawals()
    bids_by_it = (bids_df.groupby(["bidder_num", "round_num"])["bta"]
                          .apply(lambda s: set(s.astype(int)))
                          .to_dict())

    rounds = sorted(shb_wide.index.tolist())
    R = max(rounds)
    late_cut = int(np.ceil(R * (1 - late_frac)))   # rounds >= late_cut are 'late'

    # walk the SHB panel, identify flagship-loss events
    pop_by_bta = dict(zip(bta_ids, pops))
    events = []   # (i, j*, t, [j'_complementary, ...])
    prev = None
    for t in rounds:
        snap = shb_wide.loc[t]
        if prev is not None and t >= late_cut:
            common = snap.index.intersection(prev.index)
            p, c = prev[common], snap[common]
            mask = p.notna() & (p != c)

            # group by bidder so we can find each bidder's flagship loss
            for j in common[mask]:
                j = int(j)
                i = int(p[j])
                # withdrawal exclusion
                if (t, j, i) in withdraw:
                    continue
                # ≥ 3 consecutive SHB rounds before t
                ok = True
                for lag in range(1, _CONSECUTIVE_MIN + 1):
                    r = t - lag
                    past = shb_wide.loc[r].get(j, np.nan) if r in shb_wide.index else np.nan
                    if pd.isna(past) or int(past) != i:
                        ok = False; break
                if not ok:
                    continue

                # Was j i's flagship at t-1?
                portfolio = prev[prev == i].index.astype(int).tolist()
                if len(portfolio) < 2:        # nothing complementary to be stuck on
                    continue
                portfolio_pops = {b: pop_by_bta.get(b, 0.0) for b in portfolio}
                flagship = max(portfolio_pops, key=portfolio_pops.get)
                if int(flagship) != j:
                    continue

                # Complementary remaining licenses (≤ compl_km of j*).
                if j not in idx_of:
                    continue
                ji = idx_of[j]
                comp = []
                for jp in portfolio:
                    if jp == j or jp not in idx_of:
                        continue
                    if d_km[ji, idx_of[jp]] <= compl_km:
                        comp.append(jp)
                if comp:
                    events.append({"bidder": i, "flagship": j, "t": t,
                                   "complementary": comp})
        prev = snap

    # classify each (event, j') pair
    rows = []
    for ev in events:
        for jp in ev["complementary"]:
            cat = _classify(ev["bidder"], jp, ev["t"],
                            look_ahead=look_ahead, near_km=near_km,
                            idx_of=idx_of, d_km=d_km,
                            bids_by_it=bids_by_it, withdraw=withdraw)
            rows.append({"bidder": ev["bidder"], "flagship": ev["flagship"],
                         "t": ev["t"], "j_prime": jp, "category": cat})
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"  exposure_test [{tag}]: NO QUALIFYING EVENTS  "
              f"(late_frac={late_frac}, compl_km={compl_km})")
        return df

    counts = df["category"].value_counts()
    n_total = len(df)
    n_inform = int(n_total - counts.get("exited", 0))   # active-elsewhere denom
    print(f"  exposure_test [{tag}]:")
    print(f"    flagship-loss events            = {len(events):,}")
    print(f"    (event, complementary) tuples   = {n_total:,}")
    print(f"    bidders involved                = {df['bidder'].nunique()}")
    print(f"    distinct flagship licenses      = {df['flagship'].nunique()}")
    print()
    print(f"    withdrew     {counts.get('withdrew', 0):>5}  "
          f"({100*counts.get('withdrew', 0)/n_total:5.1f}%)")
    print(f"    re-engaged   {counts.get('re-engaged', 0):>5}  "
          f"({100*counts.get('re-engaged', 0)/n_total:5.1f}%)")
    print(f"    stuck        {counts.get('stuck', 0):>5}  "
          f"({100*counts.get('stuck', 0)/n_total:5.1f}%)  "
          f"← active elsewhere, silent on this fragment")
    print(f"    exited       {counts.get('exited', 0):>5}  "
          f"({100*counts.get('exited', 0)/n_total:5.1f}%)  "
          f"← uninformative (silent throughout)")
    print()
    if n_inform > 0:
        stuck_active = counts.get("stuck", 0) / n_inform
        print(f"    *** stuck-rate among active-elsewhere "
              f"= {stuck_active:.1%}  (n = {n_inform:,}) ***")

    # ── figure ───────────────────────────────────────────────────────
    cats = ["withdrew", "re-engaged", "stuck", "exited"]
    pct = [100 * counts.get(c, 0) / n_total for c in cats]
    colors = [GOLD, NAVY, "#C0392B", "#BBBBBB"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(cats[::-1], pct[::-1], color=colors[::-1],
                    edgecolor="white", linewidth=0.7)
    for bar, v in zip(bars, pct[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, family="serif",
                color=SLATE)
    ax.set_xlabel("share of (event × complementary license) tuples (%)",
                  fontsize=9, family="serif")
    ax.set_xlim(0, max(pct) * 1.15 + 5)
    style_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("serif")
    fig.tight_layout()
    out = OUT_FIG / f"fig_exposure_test_{tag}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(OUT_FIG / f"fig_exposure_test_{tag}.csv", index=False)
    print(f"    → {out}")
    return df


def main():
    print("=" * 68)
    print("Pre-specified test: exposure problem in the C-block auction.")
    print("Late phase = last 20% of rounds. Complementary ≤ 500 km of flagship.")
    print("Look-ahead = 30 rounds. Re-engaged = bid within 200 km of j'.")
    print("=" * 68)
    main_df = run(tag="main")

    # one declared robustness on the only sensitive knob: complementary radius.
    print("\n  -- robustness: tighter (250 km) and looser (1000 km) radius --")
    run(compl_km=250, tag="r250")
    run(compl_km=1000, tag="r1000")


if __name__ == "__main__":
    main()
