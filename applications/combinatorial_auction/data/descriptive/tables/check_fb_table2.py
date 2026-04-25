"""Reproduce Fox-Bajari (AEJ Micro 2013) Table 2 statistics from the auction's
own data pipeline, as a sanity check that our normalizations match the paper.

Fox-Bajari Table 2 (Winning Packages: Sample Statistics):
    Characteristic                       Mean      SD      Min     Max
    Population/distance                  0.0055   0.024    0      0.20
    Trips between markets in ATS         0.0032   0.020    0      0.182
    Total airport trips (thousands)      0.0023   0.017    0      0.150

    Correlations (sample: 85 winners in continental US):
                    Pop/dist   ATS trips
    Pop/dist        1
    ATS trips       0.97       1
    Airport trips   0.95       0.99

Also, from p.112-113 (normalization):
    mean pop_sum across winners       ≈ 0.012 (sd 0.044)
    mean elig·pop_sum across winners  ≈ 0.0046 (sd 0.030)

Quantities are computed from the exact arrays that flow into combest via
``prepare(...)``'s input_data — quadratic_item tensor (Q matrices for adjacency,
pop_centroid, air_travel, travel_survey) and id_data (obs_bundles, elig).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_raw


# ────────────────────────────────────────────────────────────────────
QUAD_NAMES = ["adjacency", "pop_centroid_00", "pop_centroid_delta2",
              "pop_centroid_delta4", "air_travel", "travel_survey"]


def per_package_complementarity(obs_bundles, Q):
    """For each row b_i of obs_bundles, compute b_i' Q b_i.
    Args:
        obs_bundles: (n_obs, n_items) 0/1
        Q:           (n_items, n_items, K)  -- K Q matrices stacked on last axis
    Returns:
        (n_obs, K)
    """
    B = obs_bundles.astype(np.float64)
    # einsum: for each obs i, compute Σ_{j,l} B[i,j] * Q[j,l,k] * B[i,l]
    return np.einsum("ij,jlk,il->ik", B, Q, B, optimize=True)


def main():
    raw = load_raw()

    # Assemble input_data exactly as the auction's estimate.py does.
    input_data, meta = prepare(
        modular_regressors       = ["elig_pop"],
        quadratic_regressors     = QUAD_NAMES,
        quadratic_id_regressors  = [],
    )
    Q         = input_data["item_data"]["quadratic"]   # (n_items, n_items, K)
    obs       = input_data["id_data"]["obs_bundles"]   # (n_obs, n_items)
    elig      = input_data["id_data"]["elig"]          # (n_obs,)  — already normalized
    pop_share = raw["bta_data"]["pop90_share"].to_numpy(float)  # (n_items,)

    # Sample: winning bidders (rowsum > 0).  Matches Fox-Bajari's "85 winning
    # bidders in the continental United States".
    is_winner  = obs.sum(axis=1) > 0
    n_winners  = int(is_winner.sum())
    obs_w      = obs[is_winner]
    elig_w     = elig[is_winner]

    print(f"n_obs={obs.shape[0]}  n_items={obs.shape[1]}  n_winners={n_winners}  "
          f"(FB Table 1 reports 85 winners)")

    # ── Per-winner complementarities (the 3 FB proxies) ─────────────
    comp = per_package_complementarity(obs_w, Q)   # (n_winners, K)
    df = pd.DataFrame(comp, columns=QUAD_NAMES)

    print("\n── Table 2 (winning packages, sample statistics) ──")
    summary = df.agg(['mean', 'std', 'min', 'max']).T
    summary.columns = ['Mean', 'SD', 'Min', 'Max']
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nFB paper values for comparison:")
    print("  Population/distance       Mean 0.0055  SD 0.024  Min 0  Max 0.20")
    print("  ATS trips                 Mean 0.0032  SD 0.020  Min 0  Max 0.182")
    print("  Airport trips (thousands) Mean 0.0023  SD 0.017  Min 0  Max 0.150")

    # ── Correlation matrix (FB's gravity + ATS + airport block) ──────
    fb_cols = ["pop_centroid_delta4", "travel_survey", "air_travel"]
    print("\n── Correlation matrix of the 3 FB proxies (winning packages) ──")
    print(df[fb_cols].corr().to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nFB paper correlations (Table 2):")
    print("                   pop/dist  ATS   airport")
    print("  pop/dist         1.00")
    print("  ATS trips        0.97      1.00")
    print("  Airport trips    0.95      0.99    1.00")

    # ── Pop-sum normalization (FB p.113 note) ───────────────────────
    pop_sum_per_winner       = obs_w @ pop_share           # Σ_{j ∈ J_i} pop_j
    elig_pop_sum_per_winner  = elig_w * pop_sum_per_winner # elig_i · Σ pop_j

    print(f"\n── Normalization check (FB p.113) ──")
    print(f"  Σ pop_j (winner mean, sd)           = {pop_sum_per_winner.mean():.4f}, "
          f"{pop_sum_per_winner.std():.4f}   (FB: 0.012, 0.044)")
    print(f"  elig·Σ pop_j (winner mean, sd)      = {elig_pop_sum_per_winner.mean():.4f}, "
          f"{elig_pop_sum_per_winner.std():.4f}   (FB: 0.0046, 0.030)")


if __name__ == "__main__":
    main()
