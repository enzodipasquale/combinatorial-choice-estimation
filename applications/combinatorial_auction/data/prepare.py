"""Covariate construction for C-block estimation.

Returns (input_data, meta):
  input_data = {
    "id_data":   {modular, obs_bundles, capacity, elig, [quadratic]},
    "item_data": {modular, quadratic, weight},
  }
  meta = {n_obs, n_items, n_btas, n_id_mod, n_item_mod, n_id_quad, n_item_quad,
          n_covariates, covariate_names}

Covariate ordering along the final axis (matches combest θ layout):
  [ modular_regressors | item_FE | quadratic_id_regressors | quadratic_regressors ]
"""
import numpy as np

from .covariates import MODULAR, QUADRATIC, QUADRATIC_ID, build
from .loaders import load_raw, build_context, filter_winners, last_round_capacity


def prepare(
    modular_regressors,
    quadratic_regressors,
    quadratic_id_regressors=(),
    *,
    winners_only=False,
    capacity_source="initial",          # "initial" | "last_round"
    upper_triangular_quadratic=False,   # zero the lower triangle of every quadratic
):
    raw = load_raw()
    ctx = build_context(raw)

    n_obs   = len(raw["bidder_data"])
    n_items = len(raw["bta_data"])

    mod_id   = build(MODULAR,      modular_regressors,      ctx)
    quad     = build(QUADRATIC,    quadratic_regressors,    ctx)
    quad_id  = build(QUADRATIC_ID, quadratic_id_regressors, ctx)

    # Optional: keep only the strict upper triangle of every quadratic covariate.
    # Combest's subproblem computes b' Q b where Q is passed as-is (Gurobi
    # accepts non-symmetric Q). Zeroing the lower triangle halves each pairwise
    # contribution to a single directed j→k term with j<k, so the bilinear form
    # becomes Σ_{j<k} Q_{jk} b_j b_k instead of Σ_{j≠k} Q_{jk} b_j b_k.
    if upper_triangular_quadratic:
        if quad is not None:
            for k in range(quad.shape[-1]):
                quad[..., k] = np.triu(quad[..., k], k=1)
        if quad_id is not None:
            triu_mask = np.triu(np.ones((n_items, n_items), dtype=bool), k=1)
            quad_id *= triu_mask[None, :, :, None]

    # id_data["modular"] must always be present for combest; quadratics are
    # optional and we let downstream code branch on `if "quadratic" in ...`.
    if mod_id is None:
        mod_id = np.zeros((n_obs, n_items, 0), dtype=np.float64)

    # Item-level fixed effects: one coefficient per item. The `-I` sign encodes
    # the convention δ_j = -θ_fe_j (utility contribution is θ_fe · (-I) · b).
    # Everything downstream (2SLS, xi, counterfactual) relies on this sign flip.
    item_mod = -np.eye(n_items, dtype=np.float64)

    id_data = {
        "modular":     mod_id,
        "capacity":    ctx["capacity"],
        "obs_bundles": ctx["c_obs_bundles"],
        "elig":        ctx["elig"],
    }
    if quad_id is not None:
        id_data["quadratic"] = quad_id

    item_data = {"modular": item_mod, "weight": ctx["weight"]}
    if quad is not None:
        item_data["quadratic"] = quad

    input_data = {"id_data": id_data, "item_data": item_data}

    # Winners-only filter and last-round capacity override.
    if winners_only:
        input_data, keep = filter_winners(input_data)
        if capacity_source == "last_round":
            input_data["id_data"]["capacity"] = last_round_capacity(
                raw["bidder_data"], keep
            )
    elif capacity_source != "initial":
        raise ValueError(
            f"capacity_source={capacity_source!r} requires winners_only=True"
        )

    # Dimensions + covariate names. Item FEs are unnamed — one per item.
    n_obs       = input_data["id_data"]["obs_bundles"].shape[0]
    n_id_mod    = mod_id.shape[-1]
    n_item_mod  = item_mod.shape[-1]
    n_id_quad   = quad_id.shape[-1] if quad_id is not None else 0
    n_item_quad = quad.shape[-1]    if quad     is not None else 0

    names = {i: n for i, n in enumerate(modular_regressors)}
    off = n_id_mod + n_item_mod
    for i, n in enumerate(quadratic_id_regressors):
        names[off + i] = n
    off += n_id_quad
    for i, n in enumerate(quadratic_regressors):
        names[off + i] = n

    meta = dict(
        n_obs=n_obs, n_items=n_items, n_btas=n_items,
        n_id_mod=n_id_mod, n_item_mod=n_item_mod,
        n_id_quad=n_id_quad, n_item_quad=n_item_quad,
        n_covariates=n_id_mod + n_item_mod + n_id_quad + n_item_quad,
        covariate_names=names,
    )
    return input_data, meta
