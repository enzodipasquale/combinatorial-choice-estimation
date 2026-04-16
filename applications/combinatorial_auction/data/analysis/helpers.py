"""Reusable data transforms for auction descriptive analysis."""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "datasets" / "114402-V1" / "Replication-Fox-and-Bajari" / "data"


def last_round_eligibility_pop(bidder_data):
    """Last-round eligibility in population units (before knapsack rescaling).

    Scales each bidder's pops_eligible by the ratio of their last active
    round's max_elig to their round-1 max_elig.
    """
    elig_df = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    swap = {190: 234, 234: 190}
    baseline = bidder_data["pops_eligible"].to_numpy().astype(float)
    n = len(bidder_data)
    out = np.copy(baseline)
    for i in range(n):
        fox_swapped = int(bidder_data.iloc[i]["bidder_num_fox"])
        fox_orig = swap.get(fox_swapped, fox_swapped)
        sub = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["max_elig"] > 0)]
        if len(sub) == 0:
            continue
        last = sub.sort_values("round_num").iloc[-1]
        r1_rows = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["round_num"] == 1)]
        if len(r1_rows) == 0:
            continue
        r1 = r1_rows["max_elig"].iloc[0]
        if r1 > 0:
            out[i] = baseline[i] * last["max_elig"] / r1
    return out


def pairwise_distances_within_packages(c_obs, dist, bidder_idx):
    """All within-package pairwise distances for the given bidder indices."""
    all_dists = []
    for i in bidder_idx:
        items = np.where(c_obs[i])[0]
        if len(items) < 2:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                all_dists.append(dist[items[a], items[b]])
    return np.array(all_dists)


def null_pairwise_distances(c_obs, dist, bidder_idx, rng, n_sims=500):
    """Random-package null: for each bidder, draw a package of the same size."""
    n_items = c_obs.shape[1]
    universe = np.arange(n_items)
    all_dists = []
    for _ in range(n_sims):
        for i in bidder_idx:
            items = np.where(c_obs[i])[0]
            k = len(items)
            if k < 2:
                continue
            fake = rng.choice(universe, size=k, replace=False)
            for a in range(len(fake)):
                for b in range(a + 1, len(fake)):
                    all_dists.append(dist[fake[a], fake[b]])
    return np.array(all_dists)
