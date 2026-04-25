"""Reproduce Fox-Bajari (AEJ Micro 2013) Table 5 values using our data pipeline.

The paper reports, at point estimates from Table 3 column 2, the deterministic
value Σ_a π_β(a, J_a) under five counterfactual allocations:

  (1) C block — 85 actual winning packages
  (2) All 480 licenses won by different bidders  (no complementarities possible)
  (3) Each MTA as a separate package (continental US has 47 MTAs in paper)
  (4) Four large regional packages (Northeast, Midwest, South, West)
  (5) Nationwide license (NextWave wins the entire United States)

Their table (Table 5):

    Allocation                           elig·Σpop  geo/dist  air trv  ATS  Total
    ──────────────────────────────────────────────────────────────────────────────
    (1) C block, 85 winning packages        0.39     0.47     0.20    0.27   0.52
    (2) All 480 licenses, different bidders 0.17     0.00     0.00    0.00   0.17
    (3) Each 47 MTAs, separate package      0.20     0.72     0.04    0.17   0.43
    (4) Four large regional licenses        0.50     0.96     0.37    0.58   0.77
    (5) Nationwide license (NextWave)       0.71     1.00     1.00    1.00   0.90

    Coefficients (Table 3 col 2): β_elig=+1, β_geo=+0.32, β_air=-0.16, β_ATS=+0.03

For each allocation we compute:
  - elig·Σpop = Σ_i elig_i · b_i · pop
  - complementarity_k = Σ_i b_i' · Q_k · b_i  (Q_k row-normalized & pop-scaled,
     so sum over ALL pairs = 1 — i.e., the nationwide package has value 1)

where b_i is the n_items indicator of package won by bidder i.  Assortative
matching: top-K bidders by initial eligibility paired in descending order of
package population.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import (
    load_raw, load_aggregation_matrix, build_context,
)
from applications.combinatorial_auction.data.descriptive.maps import xwalk


# Fox-Bajari Table 3 column 2 point estimates
BETA = {"elig": 1.0, "geo": 0.32, "air": -0.16, "ats": 0.03}

# Continental-US region assignment by MTA name (paper's hints:
#   "Midwest roughly Pittsburgh to Wichita, Washington in Northeast,
#    Oklahoma and Texas (except El Paso) in South")
REGION_BY_MTA = {
    # Northeast
    1:  "NE", 8:  "NE", 9:  "NE", 10: "NE", 35: "NE",
    # Midwest
    3:  "MW", 5:  "MW", 12: "MW", 16: "MW", 18: "MW", 19: "MW", 20: "MW",
    21: "MW", 26: "MW", 31: "MW", 32: "MW", 34: "MW", 38: "MW", 45: "MW", 46: "MW",
    # South
    6:  "S",  7:  "S",  11: "S",  13: "S",  14: "S",  15: "S",  17: "S",
    23: "S",  28: "S",  29: "S",  33: "S",  37: "S",  40: "S",  41: "S",
    43: "S",  44: "S",  48: "S",
    # West
    2:  "W",  4:  "W",  22: "W",  24: "W",  27: "W",  30: "W",  36: "W",
    39: "W",  42: "W",
}

QUAD_NAMES = ["adjacency", "pop_centroid_delta4", "air_travel", "travel_survey"]
FB_LABELS  = {"pop_centroid_delta4": "geo/dist",
              "travel_survey":       "ATS",
              "air_travel":          "air trv"}


def eval_allocation(b, pop, elig, Q):
    """b: (n_bidders, n_items) indicator of each bidder's package.
       Returns dict with eligibility piece and complementarity per Q."""
    elig_pop = float(np.sum(elig[:, None] * b * pop[None, :]))
    out = {"elig·Σpop": elig_pop}
    # Per-bidder: b_i' Q_k b_i, summed over bidders.
    comp = np.einsum("ij,jlk,il->k", b.astype(float), Q, b.astype(float), optimize=True)
    for name, v in zip(QUAD_NAMES, comp):
        out[name] = float(v)
    return out


def main():
    raw = load_raw()
    input_data, meta = prepare(
        modular_regressors       = ["elig_pop"],
        quadratic_regressors     = QUAD_NAMES,
        quadratic_id_regressors  = [],
    )
    Q = input_data["item_data"]["quadratic"]         # (n_btas, n_btas, 4)
    obs_bundles = input_data["id_data"]["obs_bundles"]
    elig        = input_data["id_data"]["elig"]
    pop         = raw["bta_data"]["pop90_share"].to_numpy(float)

    btas = raw["bta_data"]["bta"].astype(int).values
    A, mta_nums = load_aggregation_matrix(btas)      # (n_mtas, n_btas)
    n_btas = A.shape[1]

    bidder_df = raw["bidder_data"]
    n_bidders = len(bidder_df)

    # Helper: top-K bidders by elig in descending order
    sorted_bidders = np.argsort(-elig)               # bidder indices, desc by elig
    is_winner = obs_bundles.sum(axis=1) > 0
    sorted_winners = np.argsort(-np.where(is_winner, elig, -np.inf))  # winners first, by elig
    n_winners = int(is_winner.sum())

    # ─────────────────────────────────────────────────────────────
    # (1) Actual C block: 85 winning packages = obs_bundles
    # ─────────────────────────────────────────────────────────────
    row1 = eval_allocation(obs_bundles.astype(bool), pop, elig, Q)

    # ─────────────────────────────────────────────────────────────
    # (2) All 480 licenses won by different bidders.  Paper p.133:
    #     "255 bidders assortatively matched; 480-255=225 smallest-pop
    #      licenses go to the lowest-elig bidder."  Each license is a
    #     singleton package → complementarities are 0 by construction.
    #     Compute elig·Σpop directly by summing assigned-elig × pop over
    #     licenses (no bidder identity needed for the complementarity block).
    # ─────────────────────────────────────────────────────────────
    pop_order = np.argsort(-pop)                     # licenses by desc pop
    elig_assigned_per_license = np.empty(n_btas, dtype=float)
    elig_desc = np.sort(elig)[::-1]                  # all-bidder eligs, desc
    elig_min  = float(elig.min())
    for rank in range(n_btas):
        j = int(pop_order[rank])
        elig_assigned_per_license[j] = (
            elig_desc[rank] if rank < n_bidders else elig_min
        )
    row2 = {"elig·Σpop": float((elig_assigned_per_license * pop).sum())}
    for name in QUAD_NAMES:
        row2[name] = 0.0

    # ─────────────────────────────────────────────────────────────
    # (3) Each MTA as a separate package; top-N-by-elig bidders
    #     assortatively matched to MTAs in desc pop
    # ─────────────────────────────────────────────────────────────
    n_mtas = A.shape[0]
    pop_mta = A @ pop
    mta_order = np.argsort(-pop_mta)
    # Use winning bidders only (paper p.133)
    b3 = np.zeros((n_bidders, n_btas), dtype=bool)
    for rank, m_idx in enumerate(mta_order):
        i = int(sorted_winners[min(rank, n_winners - 1)])
        b3[i] = b3[i] | A[m_idx].astype(bool)
    row3 = eval_allocation(b3, pop, elig, Q)

    # ─────────────────────────────────────────────────────────────
    # (4) Four large regional packages; top-4-by-elig matched to
    #     regions in desc pop
    # ─────────────────────────────────────────────────────────────
    # Build one bundle per region from MTA→region map
    region_bundles = {}
    missing = []
    for i_mta, m_num in enumerate(mta_nums):
        r = REGION_BY_MTA.get(int(m_num))
        if r is None:
            missing.append(int(m_num)); continue
        region_bundles.setdefault(r, np.zeros(n_btas, bool))
        region_bundles[r] |= A[i_mta].astype(bool)
    if missing:
        print(f"Warning: MTAs with no region assignment: {missing}")
    region_pops = {r: float(b @ pop) for r, b in region_bundles.items()}
    regions_desc = sorted(region_bundles, key=lambda r: -region_pops[r])
    print(f"Region pops (desc): " + ", ".join(
        f"{r}={region_pops[r]:.3f}" for r in regions_desc))
    # Use winning bidders only (paper p.133)
    b4 = np.zeros((n_bidders, n_btas), dtype=bool)
    for rank, r in enumerate(regions_desc):
        i = int(sorted_winners[rank])
        b4[i] = region_bundles[r]
    row4 = eval_allocation(b4, pop, elig, Q)

    # ─────────────────────────────────────────────────────────────
    # (5) Nationwide license, NextWave wins (paper p.133)
    # ─────────────────────────────────────────────────────────────
    # NextWave is bidder_num_fox==77 per the data; confirm it's top-elig
    nw_idx = bidder_df.index[bidder_df["bidder_num_fox"] == 77].tolist()
    if not nw_idx:
        raise RuntimeError("NextWave (bidder_num_fox==77) not found in bidder_data")
    nw_idx = nw_idx[0]
    print(f"NextWave idx={nw_idx}, elig={elig[nw_idx]:.4f}, "
          f"top-elig idx={sorted_bidders[0]} elig={elig[sorted_bidders[0]]:.4f}")
    b5 = np.zeros((n_bidders, n_btas), dtype=bool)
    b5[nw_idx] = True
    row5 = eval_allocation(b5, pop, elig, Q)

    # ─────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────
    rows = [
        ("(1) C block, 85 winning packages",       row1),
        ("(2) All 480 licenses, diff bidders",      row2),
        ("(3) Each 47 MTAs, separate package",      row3),
        ("(4) Four large regional packages",        row4),
        ("(5) Nationwide license (NextWave)",       row5),
    ]

    def total(row):
        return (BETA["elig"] * row["elig·Σpop"]
                + BETA["geo"] * row["pop_centroid_delta4"]
                + BETA["air"] * row["air_travel"]
                + BETA["ats"] * row["travel_survey"])

    print(f"\n{'─'*95}")
    print(f"{'Allocation':<42}  {'elig':>6}  {'geo':>6}  {'air':>6}  {'ATS':>6}  {'total':>6}")
    print(f"{'─'*95}")
    fb_totals = [0.52, 0.17, 0.43, 0.77, 0.90]
    fb_raws = [(0.39,0.47,0.20,0.27), (0.17,0,0,0),(0.20,0.72,0.04,0.17),
               (0.50,0.96,0.37,0.58),(0.71,1.0,1.0,1.0)]
    for (lbl, row), fb_tot, fb_raw in zip(rows, fb_totals, fb_raws):
        print(f"{lbl:<42}  "
              f"{row['elig·Σpop']:>6.3f}  "
              f"{row['pop_centroid_delta4']:>6.3f}  "
              f"{row['air_travel']:>6.3f}  "
              f"{row['travel_survey']:>6.3f}  "
              f"{total(row):>6.3f}")
        print(f"{'  └── paper':<42}  "
              f"{fb_raw[0]:>6.3f}  {fb_raw[1]:>6.3f}  {fb_raw[2]:>6.3f}  "
              f"{fb_raw[3]:>6.3f}  {fb_tot:>6.3f}")
    print(f"{'─'*95}")


if __name__ == "__main__":
    main()
