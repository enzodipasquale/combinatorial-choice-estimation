"""Data utilities for zero-noise estimation variants."""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "datasets" / "114402-V1" / "Replication-Fox-and-Bajari" / "data"


def filter_winners(input_data):
    """Keep only observations with nonzero observed bundles."""
    b_obs = input_data["id_data"]["obs_bundles"]
    keep = b_obs.sum(1) > 0
    for k, v in input_data["id_data"].items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(keep):
            input_data["id_data"][k] = v[keep]
    return input_data, keep


def last_round_capacity(bidder_data, keep_mask=None):
    """Compute capacity from each bidder's last active round.

    Returns array of shape (n_kept,) in the same units as baseline capacity
    (pops_eligible // WEIGHT_ROUNDING_TICK).
    """
    from applications.combinatorial_auction.data.registries import WEIGHT_ROUNDING_TICK

    elig_df = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    swap = {190: 234, 234: 190}

    if keep_mask is not None:
        bidder_data = bidder_data.iloc[np.where(keep_mask)[0]].reset_index(drop=True)

    n = len(bidder_data)
    baseline = bidder_data["pops_eligible"].to_numpy().astype(float)
    capacity = np.zeros(n, dtype=int)

    for i in range(n):
        fox_swapped = int(bidder_data.iloc[i]["bidder_num_fox"])
        fox_orig = swap.get(fox_swapped, fox_swapped)
        sub = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["max_elig"] > 0)]
        if len(sub) == 0:
            capacity[i] = int(np.round(baseline[i] // WEIGHT_ROUNDING_TICK))
            continue
        last = sub.sort_values("round_num").iloc[-1]
        r1 = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["round_num"] == 1)]["max_elig"].iloc[0]
        if r1 > 0:
            capacity[i] = int(np.round(baseline[i] * last["max_elig"] / r1 // WEIGHT_ROUNDING_TICK))
        else:
            capacity[i] = 0

    return capacity
