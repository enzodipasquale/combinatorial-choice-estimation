import json
import numpy as np

from applications.combinatorial_auction.data.loaders import (
    load_bta_data, build_context, load_aggregation_matrix, continental_mta_nums,
)
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC
from applications.combinatorial_auction.data.prepare import _build_features, _aggregate_quadratics


def prepare_counterfactual(est_result_path_or_dict, alpha_0, alpha_1,
                           modular_regressors=None, quadratic_regressors=None,
                           quadratic_id_regressors=None, elig_scale=1.0):
    """Build MTA-level counterfactual data from C-block BTA estimation."""
    if isinstance(est_result_path_or_dict, dict):
        result = est_result_path_or_dict
    else:
        result = json.load(open(est_result_path_or_dict))
    theta = np.array(result["theta_hat"])
    n_id_mod = result["n_id_mod"]
    n_btas = result["n_btas"]
    n_id_quad_est = result.get("n_id_quad", 0)

    # read regressors from result if not provided
    spec = result.get("specification", {})
    if modular_regressors is None:
        modular_regressors = spec.get("modular", [])
    if quadratic_regressors is None:
        quadratic_regressors = spec.get("quadratic", [])
    if quadratic_id_regressors is None:
        quadratic_id_regressors = spec.get("quadratic_id", [])

    # extract estimated parameters
    # theta order: id_mod | item_mod(FE) | id_quad | item_quad
    beta = theta[:n_id_mod]
    theta_fe = theta[n_id_mod : n_id_mod + n_btas]
    gamma_id = theta[n_id_mod + n_btas : n_id_mod + n_btas + n_id_quad_est]
    gamma_item = theta[n_id_mod + n_btas + n_id_quad_est:]

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
    offset_m_no_xi = mta_sizes * alpha_0              # |m|*alpha_0 only

    # id_data: C-block bidder features aggregated to MTA level
    bta_mod = _build_features(MODULAR, modular_regressors, ctx)  # (n_obs, n_btas, n_id_mod)
    mta_mod = np.einsum('ijk,mj->imk', bta_mod, A)              # (n_obs, n_mtas, n_id_mod)

    # obs_bundles: aggregate C-block choices to MTA level
    c_obs = ctx["c_obs_bundles"].astype(float)
    mta_obs = (c_obs @ A.T > 0).astype(int)

    # item_data quadratics
    _, Q_mta = _aggregate_quadratics(ctx, quadratic_regressors, A)
    mta_weight = (A @ ctx["weight"].astype(np.float64)).astype(int)

    # id_data quadratics: elig_i * (A @ Q_bta @ A.T) for each feature
    n_id_quad = len(quadratic_id_regressors)
    qid_mta = None
    if quadratic_id_regressors:
        # each id_quad feature is elig_i * Q_item(j,l)
        # at MTA level: elig_i * (A @ Q_item @ A.T)
        elig = ctx["elig"]  # (n_obs,) normalized eligibility
        # map elig_adjacency -> adjacency, etc.
        qi2q = {name: i for i, name in enumerate(quadratic_regressors)}
        layers = []
        for name in quadratic_id_regressors:
            base_name = name.replace("elig_", "")
            q_idx = qi2q[base_name]
            # Q_mta[:,:,q_idx] is (n_mtas, n_mtas), elig is (n_obs,)
            layers.append(elig[:, None, None] * Q_mta[None, :, :, q_idx])
        qid_mta = np.stack(layers, axis=-1).astype(np.float64)  # (n_obs, n_mtas, n_mtas, n_id_quad)

    input_data = {
        "id_data": {
            "modular": mta_mod,
            "obs_bundles": mta_obs,
            "capacity": (ctx["capacity"] * elig_scale).astype(int),
        },
        "item_data": {
            "modular": -alpha_1 * np.eye(n_mtas, dtype=np.float64),
            "quadratic": Q_mta,
            "weight": mta_weight,
        },
    }
    if qid_mta is not None:
        input_data["id_data"]["quadratic"] = qid_mta

    n_obs = mta_mod.shape[0]
    n_items = n_mtas
    n_item_mod = n_mtas   # FE-style: one param per MTA = price
    n_item_quad = Q_mta.shape[-1]

    meta = {
        "n_obs": n_obs,
        "n_items": n_items,
        "n_id_mod": n_id_mod,
        "n_item_mod": n_item_mod,
        "n_id_quad": n_id_quad,
        "n_item_quad": n_item_quad,
        "n_covariates": n_id_mod + n_item_mod + n_id_quad + n_item_quad,
        "n_btas": n_btas,
        "n_mtas": n_mtas,
        "A": A,
        "offset_m": offset_m,
        "offset_m_no_xi": offset_m_no_xi,
        "beta": beta,
        "gamma_id": gamma_id,
        "gamma_item": gamma_item,
        "continental_mta_nums": continental_mta_nums(btas),
        "elig": ctx["elig"],
        "covariate_names": {},
    }

    # covariate names: id_mod | MTA FEs (unnamed) | id_quad | item_quad
    names = {i: n for i, n in enumerate(modular_regressors)}
    off = n_id_mod + n_item_mod
    for i, n in enumerate(quadratic_id_regressors):
        names[off + i] = n
    off += n_id_quad
    for i, n in enumerate(quadratic_regressors):
        names[off + i] = n
    meta["covariate_names"] = names

    return input_data, meta
