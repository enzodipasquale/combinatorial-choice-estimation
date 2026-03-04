#!/usr/bin/env python3
"""Prepare A/B block (Auction 4) data for max-score estimation.

Items = 46 continental MTAs.  Each MTA has 2 winners (one per block) due to
the FCC 45 MHz CMRS spectrum cap preventing any bidder from holding both A
and B in the same market.  obs_bundles can have up to 2 ones per column.

All regressors are BTA-level features aggregated to MTA via the BTA→MTA
aggregation matrix, following the "sum of BTA utilities" model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

AB_DIR = Path(__file__).parent
APP_DIR = AB_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.prepare_data import (
    load_raw_data, build_context, build_features,
    MODULAR, QUADRATIC, QUADRATIC_ID, WEIGHT_ROUNDING_TICK,
)
from applications.combinatorial_auction.data.prepare_data import DATA_DIR as BTA_DATA_DIR
from applications.combinatorial_auction.counterfactuals.MTA_licenses.prepare_data_counterfactual import (
    load_aggregation_matrix,
)

DATA_DIR = AB_DIR / "data"

# Regressors feasible for A/B (no assets, revenues, is_rural, hq_distance)
AB_MODULAR = ["elig_pop", "elig_percapin", "elig_hhinc35k", "elig_density", "elig_imwl"]
AB_QUADRATIC = ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]
AB_QUADRATIC_ID = ["elig_adjacency", "elig_pop_centroid_delta4"]


def load_ab_data():
    """Load A/B winning bids and bidders, return DataFrames."""
    winners = pd.read_csv(DATA_DIR / "winning_bids.csv")
    bidders = pd.read_csv(DATA_DIR / "bidders.csv")
    # drop FCC placeholder
    bidders = bidders[bidders["name"] != "FCC"].reset_index(drop=True)
    return winners, bidders


def _normalize_acct(acct):
    """Strip leading zeros for consistent matching."""
    return str(acct).lstrip("0") or "0"


def main(modular_regressors=None, quadratic_regressors=None,
         quadratic_id_regressors=None, rescale_features=True):

    modular_regressors = modular_regressors or AB_MODULAR
    quadratic_regressors = quadratic_regressors or AB_QUADRATIC
    quadratic_id_regressors = quadratic_id_regressors or AB_QUADRATIC_ID

    # ── Load BTA data (for features and matrices) ─────────────────────
    raw = load_raw_data(continental_only=True)
    ctx = build_context(raw)
    n_btas = len(raw["bta_data"])

    # ── BTA→MTA aggregation matrix ────────────────────────────────────
    continental_btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(continental_btas)  # (n_mtas_cont, n_btas)
    n_mtas = A.shape[0]  # 46 continental MTAs
    n_items = n_mtas      # items = MTAs

    # ── Continental MTA numbers (for filtering winners) ───────────────
    census = pd.read_csv(
        BTA_DATA_DIR / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    census = census[pd.to_numeric(census["BTA"], errors="coerce").notna()]
    continental_mta_nums = sorted(
        census[census["BTA"].astype(int).isin(continental_btas)]["MTA"].astype(int).unique()
    )
    mta_num_to_idx = {m: i for i, m in enumerate(continental_mta_nums)}

    # ── Load A/B data (continental only) ──────────────────────────────
    winners, bidders = load_ab_data()
    winners = winners[winners["mta_num"].isin(continental_mta_nums)].reset_index(drop=True)
    n_bidders = len(bidders)

    print(f"A/B block: {n_items} items (MTAs), {n_bidders} bidders")

    # ── Weights and capacities ────────────────────────────────────────
    bta_weight = ctx["weight"].astype(np.float64)
    mta_weight = (A @ bta_weight).astype(int)

    pop_sum = raw["bta_data"]["pop90"].sum()
    ab_elig = bidders["eligibility"].to_numpy().astype(float)
    capacity = np.round(ab_elig // WEIGHT_ROUNDING_TICK).astype(int)

    # ── obs_bundles: bidder i won MTA m if they won any block in m ───
    acct_to_idx = {_normalize_acct(a): i for i, a in enumerate(bidders["fcc_acct"])}
    obs_bundles = np.zeros((n_bidders, n_items), dtype=int)
    for _, row in winners.iterrows():
        bidder_idx = acct_to_idx.get(_normalize_acct(row["fcc_acct"]))
        if bidder_idx is None:
            continue
        mta_idx = mta_num_to_idx.get(row["mta_num"])
        if mta_idx is not None:
            obs_bundles[bidder_idx, mta_idx] = 1

    n_winners = (obs_bundles.sum(1) > 0).sum()
    n_assigned = obs_bundles.sum()
    col_sums = obs_bundles.sum(0)
    print(f"  {n_winners} winners, {n_assigned} MTA-assignments "
          f"(2 per MTA: {(col_sums == 2).sum()}, 1 per MTA: {(col_sums == 1).sum()})")

    # validate capacity
    violations = obs_bundles @ mta_weight - capacity
    viol_ids = np.where(violations > 0)[0]
    if len(viol_ids) > 0:
        print(f"  WARNING: {len(viol_ids)} capacity violations: {violations[viol_ids]}")

    # ── Aggregate BTA features to MTA level ──────────────────────────
    pop_mta = A @ ctx["pop"]
    percapin_mta = A @ ctx["percapin"]
    hhinc35k_mta = A @ ctx["hhinc35k"]
    density_mta = A @ ctx["density"]
    imwl_mta = A @ ctx["imwl"]

    # ── MTA prices (average of A/B block winning bids, normalized) ──
    mta_avg_price = winners.groupby("mta_num")["price"].mean()
    price_mta = np.array([mta_avg_price.get(m, 0) for m in continental_mta_nums])
    price_mta_norm = price_mta / price_mta.max()

    # ── Build modular regressors (n_bidders × n_items × n_features) ──
    elig_norm = ab_elig / pop_sum

    modular_registry = {
        "elig_pop":      lambda: elig_norm[:, None] * pop_mta[None, :],
        "elig_percapin": lambda: elig_norm[:, None] * percapin_mta[None, :],
        "elig_hhinc35k": lambda: elig_norm[:, None] * hhinc35k_mta[None, :],
        "elig_density":  lambda: elig_norm[:, None] * density_mta[None, :],
        "elig_imwl":     lambda: elig_norm[:, None] * imwl_mta[None, :],
        "elig_price":    lambda: elig_norm[:, None] * price_mta_norm[None, :],
    }

    mod_layers = [modular_registry[name]() for name in modular_regressors]
    modular_features = np.stack(mod_layers, axis=-1).astype(np.float64)

    # ── Build quadratic regressors (n_mtas × n_mtas × n_features) ────
    bta_quad = build_features(QUADRATIC, quadratic_regressors, ctx, rescale=False)
    # aggregate: Q_mta[m, m'] = sum_{j in m, j' in m'} Q_bta[j, j']
    n_qfeat = bta_quad.shape[-1]
    Q_mta = np.stack([A @ bta_quad[:, :, k] @ A.T for k in range(n_qfeat)], axis=-1)

    # extract diagonal (within-MTA BTA complementarities → item modular)
    diag_quad = np.array([np.diag(Q_mta[:, :, k]) for k in range(n_qfeat)]).T
    for k in range(n_qfeat):
        np.fill_diagonal(Q_mta[:, :, k], 0)

    # ── Build quadratic_id regressors (n_bidders × n_items × n_items × n_feat)
    quad_id_features = None
    if quadratic_id_regressors:
        quad_name_to_idx = {name: i for i, name in enumerate(quadratic_regressors)}
        qid_layers = []
        for name in quadratic_id_regressors:
            base_quad_name = name.replace("elig_", "")
            if base_quad_name not in quad_name_to_idx:
                raise ValueError(f"quad_id '{name}' requires quadratic '{base_quad_name}'")
            k = quad_name_to_idx[base_quad_name]
            q_slice = Q_mta[:, :, k]
            qid_layers.append(elig_norm[:, None, None] * q_slice[None, :, :])
        quad_id_features = np.stack(qid_layers, axis=-1).astype(np.float64)

    # ── Rescale ───────────────────────────────────────────────────────
    if rescale_features:
        for arr in [modular_features, Q_mta]:
            if arr.size > 0:
                spatial_axes = tuple(range(arr.ndim - 1))
                stds = arr.std(spatial_axes, keepdims=True)
                stds[stds == 0] = 1.0
                arr /= stds
        if quad_id_features is not None and quad_id_features.size > 0:
            spatial_axes = tuple(range(quad_id_features.ndim - 1))
            stds = quad_id_features.std(spatial_axes, keepdims=True)
            stds[stds == 0] = 1.0
            quad_id_features /= stds
        # rescale diag_quad with same factor as Q_mta per feature
        if diag_quad.size > 0:
            q_stds = Q_mta.std(axis=(0, 1))
            q_stds[q_stds == 0] = 1.0
            diag_quad /= q_stds

    # ── Item modular: MTA FE + quadratic diagonals ───────────────────
    item_modular = np.hstack([-np.eye(n_items), diag_quad])

    # ── Assemble output ──────────────────────────────────────────────
    input_data = {
        "id_data": {
            "modular": modular_features,
            "capacity": capacity,
            "obs_bundles": obs_bundles,
        },
        "item_data": {
            "modular": item_modular,
            "quadratic": Q_mta,
            "weight": mta_weight,
        },
    }
    if quad_id_features is not None:
        input_data["id_data"]["quadratic"] = quad_id_features

    # ── Print stats ──────────────────────────────────────────────────
    n_id_mod = modular_features.shape[-1]
    n_item_mod = item_modular.shape[-1]
    print(f"  id modular:    {modular_features.shape}")
    print(f"  item modular:  {item_modular.shape} ({n_items} FE + {diag_quad.shape[1]} diag)")
    print(f"  quadratic:     {Q_mta.shape}")
    if quad_id_features is not None:
        print(f"  id quadratic:  {quad_id_features.shape}")
    print(f"  weights:       [{mta_weight.min()}, {mta_weight.max()}]")
    print(f"  capacities:    [{capacity.min()}, {capacity.max()}]")

    return input_data, {
        "n_mtas": n_mtas,
        "A": A,
        "continental_mta_nums": continental_mta_nums,
    }


if __name__ == "__main__":
    main()
