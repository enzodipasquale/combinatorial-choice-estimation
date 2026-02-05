#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent.parent))
from data.prepare_data import main as prepare_bta_data, load_raw_data, DATA_DIR, NON_CONTINENTAL_BTAS

CENSUS_BTA_PATH = DATA_DIR / "cntysv2000_census-bta-may2009.csv"


def load_aggregation_matrix(continental_btas: np.ndarray) -> np.ndarray:
    df = pd.read_csv(CENSUS_BTA_PATH, encoding="latin-1")
    df = df.assign(bta=pd.to_numeric(df["BTA"], errors="coerce"), mta=pd.to_numeric(df["MTA"], errors="coerce")).dropna()
    m = df[df["bta"].isin(continental_btas)]
    bta_col = {b: i for i, b in enumerate(continental_btas)}
    mta_ids = sorted(m["mta"].unique().astype(int).tolist())
    mta_row = {mta: i for i, mta in enumerate(mta_ids)}
    A = np.zeros((len(mta_ids), len(continental_btas)))
    for _, r in m.iterrows():
        A[mta_row[int(r["mta"])], bta_col[int(r["bta"])]] = 1
    return A


def aggregate_to_mta(bta_data: dict, continental_btas: np.ndarray) -> dict:
    A = load_aggregation_matrix(continental_btas)
    n_mtas, n_btas = A.shape

    # Weights: sum populations
    weights_mta = (A @ bta_data["item_data"]["weight"]).astype(int)
    
    # Quadratic: A @ Q @ A.T for each feature
    Q_bta = bta_data["item_data"]["quadratic"]
    Q_mta = np.stack([A @ Q_bta[:,:,k] @ A.T for k in range(Q_bta.shape[-1])], axis=-1)
    diagonals = np.array([np.diag(Q_mta[:,:,k]) for k in range(Q_mta.shape[-1])]).T
    for k in range(Q_mta.shape[-1]):
        np.fill_diagonal(Q_mta[:, :, k], 0)

    # Modular agent: recompute with MTA weights
    capacities = bta_data["id_data"]["capacity"]
    w_sum = weights_mta.sum()
    modular_mta = ((capacities[:,None] / w_sum) * (weights_mta[None,:] / w_sum))[:,:,None]

    item_modular = np.hstack([-np.eye(n_mtas), diagonals])
    obs_bundles = np.zeros((len(capacities), n_mtas), dtype=int)
    n_assign = min(len(capacities), n_mtas)
    obs_bundles[:n_assign, :n_assign] = np.eye(n_assign)

    return {
        "id_data": {
            "modular": modular_mta,
            "capacity": capacities,
            "obs_bundles": obs_bundles,
        },
        "item_data": {
            "modular": item_modular,
            "quadratic": Q_mta,
            "weight": weights_mta,
        },
    }


def main(delta=4, winners_only=False, continental_only=True, include_adjacency=False):
    bta_data = prepare_bta_data(
        delta=delta, winners_only=winners_only, hq_distance=False,
        continental_only=continental_only, include_adjacency=include_adjacency,
    )

    raw = load_raw_data()
    bta_ids = raw["bta_data"]["bta"].values
    if continental_only:
        bta_ids = bta_ids[~np.isin(bta_ids, list(NON_CONTINENTAL_BTAS))]

    mta_data = aggregate_to_mta(bta_data, bta_ids)

    print(f"MTA data: {mta_data['item_data']['weight'].shape[0]} items, {len(mta_data['id_data']['capacity'])} bidders")
    return mta_data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--delta", "-d", type=int, default=4)
    p.add_argument("--winners-only", "-w", action="store_true")
    p.add_argument("--adjacency", action="store_true")
    args = p.parse_args()
    main(delta=args.delta, winners_only=args.winners_only, include_adjacency=args.adjacency)