import json
import numpy as np
from pathlib import Path

SPECS_DIR = Path(__file__).parent.parent.parent
APP_DIR = SPECS_DIR.parent

from applications.combinatorial_auction.data.loaders import (
    load_bta_data, build_context, load_aggregation_matrix, continental_mta_nums,
)
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC
from applications.combinatorial_auction.data.prepare import _build_features, _aggregate_quadratics


def prepare_counterfactual(est_result_path, alpha_0, alpha_1,
                           modular_regressors, quadratic_regressors):
    """Build MTA-level counterfactual data from C-block BTA estimation."""
    result = json.load(open(est_result_path))
    theta = np.array(result["theta_hat"])
    n_id_mod = result["n_id_mod"]
    n_btas = result["n_btas"]

    # extract estimated parameters
    beta = theta[:n_id_mod]                          # id_modular (elig_pop, ...)
    theta_fe = theta[n_id_mod : n_id_mod + n_btas]   # item FEs
    gamma = theta[n_id_mod + n_btas:]                 # quadratic params

    # recover delta and xi
    raw = load_bta_data()
    ctx = build_context(raw)
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    delta = -theta_fe                                 # delta_j = -theta_j^FE
    xi = delta - alpha_0 + alpha_1 * price_bta        # xi_j = delta_j - alpha_0 + alpha_1*p_j

    # aggregation
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_mtas = A.shape[0]
    mta_sizes = A.sum(1)                              # |m|
    xi_m = A @ xi
    offset_m = mta_sizes * alpha_0 + xi_m             # |m|*alpha_0 + A@xi

    # id_data: C-block bidder features aggregated to MTA level
    bta_mod = _build_features(MODULAR, modular_regressors, ctx)  # (n_obs, n_btas, n_id_mod)
    mta_mod = np.einsum('ijk,mj->imk', bta_mod, A)              # (n_obs, n_mtas, n_id_mod)

    # obs_bundles: aggregate C-block choices to MTA level
    c_obs = ctx["c_obs_bundles"].astype(float)
    mta_obs = (c_obs @ A.T > 0).astype(int)

    # item_data
    _, Q_mta = _aggregate_quadratics(ctx, quadratic_regressors, A)
    mta_weight = (A @ ctx["weight"].astype(np.float64)).astype(int)

    input_data = {
        "id_data": {
            "modular": mta_mod,
            "obs_bundles": mta_obs,
            "capacity": ctx["capacity"],
        },
        "item_data": {
            "modular": -alpha_1 * np.eye(n_mtas, dtype=np.float64),
            "quadratic": Q_mta,
            "weight": mta_weight,
        },
    }

    n_obs = mta_mod.shape[0]
    n_items = n_mtas
    n_item_mod = n_mtas   # FE-style: one param per MTA = price
    n_id_quad = 0
    n_item_quad = Q_mta.shape[-1]

    meta = {
        "n_obs": n_obs,
        "n_items": n_items,
        "n_id_mod": n_id_mod,
        "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad,
        "n_item_quad": n_item_quad,
        "n_covariates": n_id_mod + n_item_mod + n_item_quad,
        "n_btas": n_btas,
        "n_mtas": n_mtas,
        "A": A,
        "offset_m": offset_m,
        "beta": beta,
        "gamma": gamma,
        "continental_mta_nums": continental_mta_nums(btas),
        "covariate_names": {},
    }

    # covariate names: id_modular, then MTA prices (unnamed FEs), then quadratics
    names = {i: n for i, n in enumerate(modular_regressors)}
    off = n_id_mod + n_item_mod
    for i, n in enumerate(quadratic_regressors):
        names[off + i] = n
    meta["covariate_names"] = names

    return input_data, meta
