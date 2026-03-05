#!/usr/bin/env python3
"""Prepare stacked data for joint C-block + A/B-block estimation.

Items 0..n_btas-1  = BTA licenses (C-block agents only)
Items n_btas..end  = MTA licenses (A/B agents only)

Enforced by per-agent item_mask.  Shared structural params, separate FEs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

JOINT_DIR = Path(__file__).parent
APP_DIR = JOINT_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.prepare_data import (
    load_raw_data, build_context, build_features, load_aggregation_matrix,
    MODULAR, QUADRATIC, QUADRATIC_ID, WEIGHT_ROUNDING_TICK, DATA_DIR as BTA_DATA_DIR,
)
from applications.combinatorial_auction.block_ab.prepare_data import load_ab_data, _normalize_acct

# ── Helpers ──────────────────────────────────────────────────────────────

def _continental_mta_nums(continental_btas):
    """Sorted list of continental MTA numbers from census-BTA mapping."""
    census = pd.read_csv(BTA_DATA_DIR / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    census = census[pd.to_numeric(census["BTA"], errors="coerce").notna()]
    return sorted(census[census["BTA"].astype(int).isin(continental_btas)]["MTA"].astype(int).unique())


def _build_ab_obs_bundles(winners, bidders, mta_num_to_idx):
    """Build (n_bidders, n_mtas) observed bundle matrix from A/B winning bids."""
    acct_to_idx = {_normalize_acct(a): i for i, a in enumerate(bidders["fcc_acct"])}
    obs = np.zeros((len(bidders), len(mta_num_to_idx)), dtype=int)
    for _, row in winners.iterrows():
        bi = acct_to_idx.get(_normalize_acct(row["fcc_acct"]))
        mi = mta_num_to_idx.get(row["mta_num"])
        if bi is not None and mi is not None:
            obs[bi, mi] = 1
    return obs


def _stack_block_diag(c_arr, ab_arr, n_obs_c, n_obs_ab, n_btas):
    """Zero-pad and stack two block-diagonal arrays along obs and item dims."""
    shape = list(c_arr.shape)
    # Replace obs dim with total, item dims with n_items
    n_items = n_btas + ab_arr.shape[1]
    ndim = c_arr.ndim
    if ndim == 3:  # (obs, items, feat)
        out = np.zeros((n_obs_c + n_obs_ab, n_items, shape[-1]), dtype=np.float64)
        out[:n_obs_c, :n_btas] = c_arr
        out[n_obs_c:, n_btas:] = ab_arr
    elif ndim == 4:  # (obs, items, items, feat)
        out = np.zeros((n_obs_c + n_obs_ab, n_items, n_items, shape[-1]), dtype=np.float64)
        out[:n_obs_c, :n_btas, :n_btas] = c_arr
        out[n_obs_c:, n_btas:, n_btas:] = ab_arr
    return out

# ── Main ─────────────────────────────────────────────────────────────────

def main(modular_regressors=None, quadratic_regressors=None, quadratic_id_regressors=None):
    modular_regressors = modular_regressors or ["elig_pop"]
    quadratic_regressors = quadratic_regressors or ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]
    quadratic_id_regressors = quadratic_id_regressors or ["elig_adjacency", "elig_pop_centroid_delta4"]

    # ── C-block BTA data ─────────────────────────────────────────────
    raw = load_raw_data(continental_only=True)
    ctx = build_context(raw)
    n_btas, n_obs_c = len(raw["bta_data"]), len(raw["bidder_data"])
    pop_sum = raw["bta_data"]["pop90"].sum()

    c_mod = build_features(MODULAR, modular_regressors, ctx)            # (n_obs_c, n_btas, n_mod)
    c_quad = build_features(QUADRATIC, quadratic_regressors, ctx)       # (n_btas, n_btas, n_qfeat)
    c_qid = build_features(QUADRATIC_ID, quadratic_id_regressors, ctx) if quadratic_id_regressors else None

    # ── BTA -> MTA aggregation ───────────────────────────────────────
    continental_btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(continental_btas)                       # (n_mtas, n_btas)
    n_mtas = A.shape[0]
    mta_nums = _continental_mta_nums(continental_btas)
    mta_num_to_idx = {m: i for i, m in enumerate(mta_nums)}

    # MTA quadratics: A @ Q_bta @ A^T (diagonal intact = within-MTA synergy)
    n_qfeat = c_quad.shape[-1]
    Q_mta = np.stack([A @ c_quad[:, :, k] @ A.T for k in range(n_qfeat)], axis=-1)

    # MTA-level BTA features
    mta_feats = {name: A @ ctx[name] for name in ["pop", "percapin", "hhinc35k", "density", "imwl"]}

    # ── A/B block data ───────────────────────────────────────────────
    winners, bidders = load_ab_data()
    winners = winners[winners["mta_num"].isin(mta_nums)].reset_index(drop=True)
    n_obs_ab = len(bidders)

    ab_elig = bidders["eligibility"].to_numpy().astype(float)
    ab_elig_norm = ab_elig / pop_sum
    ab_capacity = np.round(ab_elig // WEIGHT_ROUNDING_TICK).astype(int)
    mta_weight = (A @ ctx["weight"].astype(np.float64)).astype(int)

    # A/B modular: elig * aggregated_feature at MTA level
    mta_mod_registry = {f"elig_{k}": (lambda f=v: ab_elig_norm[:, None] * f[None, :]) for k, v in mta_feats.items()}
    ab_mod = np.stack([mta_mod_registry[name]() for name in modular_regressors], axis=-1).astype(np.float64)

    # A/B quadratic_id: elig * Q_mta slices
    ab_qid = None
    if quadratic_id_regressors:
        qi2q = {name: i for i, name in enumerate(quadratic_regressors)}
        layers = []
        for name in quadratic_id_regressors:
            base = name.replace("elig_", "")
            assert base in qi2q, f"quad_id '{name}' requires quadratic '{base}'"
            layers.append(ab_elig_norm[:, None, None] * Q_mta[None, :, :, qi2q[base]])
        ab_qid = np.stack(layers, axis=-1).astype(np.float64)

    ab_obs = _build_ab_obs_bundles(winners, bidders, mta_num_to_idx)

    # ── Stack into joint arrays ──────────────────────────────────────
    n_items = n_btas + n_mtas
    n_obs = n_obs_c + n_obs_ab

    input_data = {
        "id_data": {
            "modular":     _stack_block_diag(c_mod, ab_mod, n_obs_c, n_obs_ab, n_btas),
            "obs_bundles":  np.zeros((n_obs, n_items), dtype=int),
            "item_mask":    np.zeros((n_obs, n_items), dtype=np.int32),
            "capacity":     np.concatenate([ctx["capacity"], ab_capacity]),
        },
        "item_data": {
            "modular":   -np.eye(n_items, dtype=np.float64),
            "quadratic":  np.zeros((n_items, n_items, n_qfeat), dtype=np.float64),
            "weight":     np.concatenate([ctx["weight"], mta_weight]),
        },
    }
    if c_qid is not None:
        input_data["id_data"]["quadratic"] = _stack_block_diag(c_qid, ab_qid, n_obs_c, n_obs_ab, n_btas)

    # Fill block-diagonal structures
    d = input_data
    d["id_data"]["obs_bundles"][:n_obs_c, :n_btas] = ctx["matching"]
    d["id_data"]["obs_bundles"][n_obs_c:, n_btas:] = ab_obs
    d["id_data"]["item_mask"][:n_obs_c, :n_btas] = 1
    d["id_data"]["item_mask"][n_obs_c:, n_btas:] = 1
    d["item_data"]["quadratic"][:n_btas, :n_btas] = c_quad
    d["item_data"]["quadratic"][n_btas:, n_btas:] = Q_mta

    # ── Diagnostics ──────────────────────────────────────────────────
    print(f"\nJoint data: {n_obs} obs ({n_obs_c} C + {n_obs_ab} AB), "
          f"{n_items} items ({n_btas} BTA + {n_mtas} MTA)")
    for k, v in {**d["id_data"], **d["item_data"]}.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            print(f"  {k:18s} {v.shape}")

    c_viol = (ctx["matching"] @ ctx["weight"] > ctx["capacity"]).sum()
    ab_viol = (ab_obs @ mta_weight > ab_capacity).sum()
    if c_viol + ab_viol:
        print(f"  WARNING: {c_viol} C + {ab_viol} AB capacity violations")

    meta = {"n_btas": n_btas, "n_mtas": n_mtas, "n_obs_c": n_obs_c,
            "n_obs_ab": n_obs_ab, "A": A, "continental_mta_nums": mta_nums}
    return input_data, meta


if __name__ == "__main__":
    main()
