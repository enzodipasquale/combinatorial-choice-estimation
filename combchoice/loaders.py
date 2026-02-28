import numpy as np
import pandas as pd


def load_market_data(product_data, bundle_dict=None, availability_data=None, covariates=None):
    obs = product_data
    n_obs = len(obs)

    has_product = "product_id" in obs.columns
    has_bundle = "bundle_id" in obs.columns
    if has_product == has_bundle:
        raise ValueError("Provide exactly one of 'product_id' or 'bundle_id'.")
    id_col = "product_id" if has_product else "bundle_id"

    for col in ("market_id", "share", "market_size"):
        if col not in obs.columns:
            raise ValueError(f"Missing required column: '{col}'")

    if has_product and bundle_dict is None:
        items, inv = np.unique(obs["product_id"].values, return_inverse=True)
        n_items = len(items)
        obs_bundles = np.eye(n_items, dtype=bool)[inv]
    else:
        if bundle_dict is None:
            raise ValueError("bundle_id requires bundle_dict.")
        bd = bundle_dict.set_index("bundle_id")
        n_items = bd.shape[1]
        obs_bundles = bd.loc[obs[id_col].values].values.astype(bool)

    obs_quantity = (obs["share"].values * obs["market_size"].values).astype(np.float64)

    _, market_ids = np.unique(obs["market_id"].values, return_inverse=True)
    market_ids = market_ids.astype(np.int64)

    if availability_data is not None:
        _, avail_inv = np.unique(availability_data["market_id"].values, return_inverse=True)
        avail_matrix = availability_data.drop(columns="market_id").values.astype(bool)
        avail_by_market = {i: avail_matrix[avail_inv == i][0] for i in np.unique(avail_inv)}
        available_items = np.array([avail_by_market[m] for m in market_ids], dtype=bool)
    else:
        available_items = np.zeros((n_obs, n_items), dtype=bool)
        for m in np.unique(market_ids):
            available_items[market_ids == m] = obs_bundles[market_ids == m].any(axis=0)

    raw_cols = list(covariates) if covariates is not None else [
        c for c in obs.columns if c not in {"market_id", id_col, "share", "market_size"}]

    id_data = {"obs_bundles": obs_bundles, "obs_quantity": obs_quantity,
               "market_id": market_ids, "available_items": available_items}

    if raw_cols:
        df_cols = [c for c in raw_cols if c != "constant"]
        data = obs[df_cols].values.astype(np.float64) if df_cols else np.empty((n_obs, 0))
        if "constant" in raw_cols:
            data = np.insert(data, raw_cols.index("constant"), 1.0, axis=1)
        id_data["data"] = data
        id_data["col_names"] = raw_cols

    return id_data, n_obs, n_items
