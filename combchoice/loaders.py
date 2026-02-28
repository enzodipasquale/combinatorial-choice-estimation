import numpy as np
import pandas as pd


def load_market_data(product_data, bundle_dict=None, availability_data=None):
    obs = product_data
    n_obs = len(obs)

    has_product = "product_id" in obs.columns
    has_bundle = "bundle_id" in obs.columns
    if has_product == has_bundle:
        raise ValueError("Provide exactly one of 'product_id' or 'bundle_id' in observations.")
    id_col = "product_id" if has_product else "bundle_id"
    single_item = has_product and bundle_dict is None

    for col in ("market_id", "share", "market_size"):
        if col not in obs.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # obs_bundles
    if single_item:
        items = np.sort(obs["product_id"].unique())
        item_to_idx = {v: i for i, v in enumerate(items)}
        n_items = len(items)
        obs_bundles = np.zeros((n_obs, n_items), dtype=bool)
        for i, pid in enumerate(obs["product_id"].values):
            obs_bundles[i, item_to_idx[pid]] = True
    else:
        if bundle_dict is None:
            raise ValueError("bundle_id in observations requires bundle_dict.")
        bd = bundle_dict.set_index("bundle_id")
        n_items = bd.shape[1]
        bundle_map = {bid: row.values.astype(bool) for bid, row in bd.iterrows()}
        obs_bundles = np.zeros((n_obs, n_items), dtype=bool)
        for i, bid in enumerate(obs[id_col].values):
            obs_bundles[i] = bundle_map[bid]

    obs_quantity = (obs["share"].values * obs["market_size"].values).astype(np.float64)

    structural = {"market_id", id_col, "share", "market_size"}
    raw_cols = [c for c in obs.columns if c not in structural]
    market_ids = obs["market_id"].values

    # available_items
    if availability_data is not None:
        ma = availability_data.set_index("market_id")
        avail_by_market = {mid: row.values.astype(bool) for mid, row in ma.iterrows()}
    else:
        avail_by_market = {}
        for mid in np.unique(market_ids):
            avail_by_market[mid] = obs_bundles[market_ids == mid].any(axis=0)
    available_items = np.array([avail_by_market[mid] for mid in market_ids], dtype=bool)

    id_data = {
        "obs_bundles": obs_bundles,
        "obs_quantity": obs_quantity,
        "market_id": market_ids,
        "available_items": available_items,
    }
    if raw_cols:
        id_data["raw_obs_data"] = obs[raw_cols].values.astype(np.float64)
        id_data["raw_obs_data_names"] = raw_cols

    return (
        {"id_data": id_data, "item_data": {}},
        {"n_obs": n_obs, "n_items": n_items},
    )
