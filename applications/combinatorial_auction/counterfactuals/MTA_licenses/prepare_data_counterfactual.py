#!/usr/bin/env python3
# Aggregate BTA-level estimation data to MTA-level for counterfactual.

import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
APP_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.prepare_data import (
    main as prepare_bta_data,
    load_raw_data,
    DATA_DIR,
)

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


def aggregate_to_mta(bta_data, continental_btas, n_id_mod=None, xi_hat=None):
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

    # Agent modular: aggregate BTA features to MTA by summing within each MTA
    # id_mod_bta is (n_obs, n_btas, n_covariates); we sum over BTAs in each MTA
    id_mod_bta = bta_data["id_data"]["modular"]  # (n_obs, n_btas, n_feat_total)
    if n_id_mod is not None:
        id_mod_bta = id_mod_bta[:, :, :n_id_mod]
    # modular_mta[i, m, k] = Σ_{j∈MTA(m)} id_mod_bta[i, j, k]
    modular_mta = np.einsum('ijk,mj->imk', id_mod_bta, A)

    capacities = bta_data["id_data"]["capacity"]

    # Item modular: FE + quadratic diagonals + optional ξ̂ column
    item_modular_parts = [-np.eye(n_mtas), diagonals]
    if xi_hat is not None:
        item_modular_parts.append((A @ xi_hat).reshape(-1, 1))
    item_modular = np.hstack(item_modular_parts)

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


def main(winners_only=False, continental_only=True, rescale_features=False,
         modular_regressors=None, quadratic_regressors=None, fe_decomp=None):
    bta_data = prepare_bta_data(
        winners_only=winners_only,
        continental_only=continental_only,
        rescale_features=rescale_features,
        modular_regressors=modular_regressors,
        quadratic_regressors=quadratic_regressors,
        quadratic_id_regressors=None,
    )

    raw = load_raw_data(continental_only)
    bta_ids = raw["bta_data"]["bta"].values
    n_id_mod = len(modular_regressors) if modular_regressors else None

    # ξ̂ column: (ξ̂_j - α₀)/α₁ per BTA → MTA FE capture only prices
    xi_hat = None
    if fe_decomp is not None:
        alpha_0, alpha_1 = fe_decomp["alpha_0"], fe_decomp["alpha_1"]
        xi_hat = (np.array(fe_decomp["xi_hat"]) - alpha_0) / alpha_1

    mta_data = aggregate_to_mta(bta_data, bta_ids, n_id_mod=n_id_mod,
                                xi_hat=xi_hat)

    n_mtas = mta_data['item_data']['weight'].shape[0]
    n_mod = mta_data['item_data']['modular'].shape[1]
    print(f"MTA data: {n_mtas} items, {len(mta_data['id_data']['capacity'])} bidders, "
          f"item_modular cols={n_mod}")
    return mta_data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--winners-only", "-w", action="store_true")
    p.add_argument("--rescale-features", action="store_true")
    p.add_argument("--quadratic", nargs="*", default=None,
                   help="Quadratic regressors to include")
    args = p.parse_args()
    main(winners_only=args.winners_only, rescale_features=args.rescale_features,
         quadratic_regressors=args.quadratic)
