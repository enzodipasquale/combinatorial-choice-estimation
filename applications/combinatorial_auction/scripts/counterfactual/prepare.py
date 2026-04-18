"""BTA → MTA aggregation for the counterfactual.

Everything here consumes the BTA-level `input_data` produced by
`data.prepare()` and the BTA-level `context`; no covariate is re-built from the
registry, so the registry is the single source of truth.

The counterfactual model has one observation per MTA and lets combest re-solve
for the optimal MTA "prices" (item-FE coefficients), holding fixed every
(β, γ_id, γ_item) identified in the BTA stage. Two offset variants:

    offset_m         = |m|·α₀ + A·(Z_j'γ_demand) + A·ξ_j         (with_xi)
    offset_m_no_xi   = |m|·α₀ + A·(Z_j'γ_demand)                 (no_xi)
"""
import numpy as np

from ...data.prepare import prepare
from ...data.loaders import load_raw, build_context, load_aggregation_matrix


def _aggregate(input_data_bta, A):
    """Aggregate BTA-level arrays to MTA level via the incidence matrix A."""
    mod_bta  = input_data_bta["id_data"]["modular"]                # (n_obs, n_bta, K_mod)
    qid_bta  = input_data_bta["id_data"].get("quadratic")          # (n_obs, n_bta, n_bta, K_qid) or None
    Qb       = input_data_bta["item_data"].get("quadratic")        # (n_bta, n_bta, K_quad) or None
    weight_b = input_data_bta["item_data"]["weight"]               # (n_bta,)

    # (n_obs, n_mta, K_mod): mta_mod[i,m,k] = Σ_j A[m,j] · bta_mod[i,j,k]
    mod_mta = np.einsum("ijk,mj->imk", mod_bta, A)

    # (n_mta, n_mta, K_quad): Q_mta[m,n,k] = Σ_{j,l} A[m,j] A[n,l] Q[j,l,k]
    Q_mta = (np.einsum("mj,jlk,nl->mnk", A, Qb, A) if Qb is not None else None)

    # (n_obs, n_mta, n_mta, K_qid) — each qid covariate is elig_i * (A Q A^T)
    # By construction (see data/covariates.py) qid_bta[i,j,l,k] = elig_i * Q_k[j,l].
    # So the aggregated form is qid_mta[i,m,n,k] = elig_i * Q_mta[m,n,k].
    # Implemented as: aggregate the first-layer quadratic (per i) via A.
    qid_mta = None
    if qid_bta is not None:
        qid_mta = np.einsum("mj,ijlk,nl->imnk", A, qid_bta, A)

    return mod_mta, qid_mta, Q_mta, (A @ weight_b.astype(float)).astype(int)


def _unpack_theta(theta, n_id_mod, n_btas, n_id_quad):
    """Split the BTA θ vector into (β, θ_fe, γ_id, γ_item)."""
    th = np.asarray(theta)
    beta       = th[:n_id_mod]
    theta_fe   = th[n_id_mod:n_id_mod + n_btas]
    gamma_id   = th[n_id_mod + n_btas:n_id_mod + n_btas + n_id_quad]
    gamma_item = th[n_id_mod + n_btas + n_id_quad:]
    return beta, theta_fe, gamma_id, gamma_item


def prepare_counterfactual(theta, app, *, alpha_0, alpha_1,
                           demand_controls=None, elig_scale=1.0):
    """Build the MTA-level input_data and structural quantities from θ.

    Args:
        theta: BTA-level point estimate, length n_id_mod + n_btas + n_id_quad + n_item_quad.
        app: the `application` block from the estimation config.
        alpha_0, alpha_1: from 2SLS on δ.
        demand_controls: dict {var_name: coef} from 2SLS (None or {} for simple IV).
        elig_scale: multiplier on MTA-level capacity.

    Returns (input_data_mta, meta, cf) where cf is a dict:
        A, offset_m, offset_m_no_xi, beta, gamma_id, gamma_item, xi, delta.
    """
    mod_names  = app.get("modular_regressors", [])
    quad_names = app.get("quadratic_regressors", [])
    qid_names  = app.get("quadratic_id_regressors", [])

    # Single source of truth: BTA-level covariates.
    bta_data, bta_meta = prepare(
        modular_regressors       = mod_names,
        quadratic_regressors     = quad_names,
        quadratic_id_regressors  = qid_names,
    )
    raw = load_raw()
    ctx = build_context(raw)
    btas = raw["bta_data"]["bta"].astype(int).values
    A, mta_nums = load_aggregation_matrix(btas)
    n_btas, n_mtas = A.shape[1], A.shape[0]
    n_id_mod  = bta_meta["n_id_mod"]
    n_id_quad = bta_meta["n_id_quad"]

    beta, theta_fe, gamma_id, gamma_item = _unpack_theta(
        theta, n_id_mod, n_btas, n_id_quad
    )

    # Structural delta and xi at BTA level.
    price_bta = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    delta = -theta_fe
    controls_bta = np.zeros(n_btas)
    if demand_controls:
        for v, c in demand_controls.items():
            controls_bta += c * raw["bta_data"][v].to_numpy(dtype=float)
    xi = delta - alpha_0 - controls_bta + alpha_1 * price_bta

    # MTA-level offset.
    mta_sizes = A.sum(1)
    A_controls = A @ controls_bta
    offset_m       = mta_sizes * alpha_0 + A_controls + A @ xi
    offset_m_no_xi = mta_sizes * alpha_0 + A_controls

    # Aggregate BTA-level covariates to MTA level.
    mod_mta, qid_mta, Q_mta, mta_weight = _aggregate(bta_data, A)

    # Synthetic obs_bundles: each MTA must appear exactly once across the rows so
    # that every MTA item-FE (= price) has a non-zero coefficient in the combest
    # master LP. The assignment of MTAs to rows is arbitrary — price solutions
    # are invariant to it (verified empirically).
    n_obs_total = mod_mta.shape[0]
    mta_obs = np.zeros((n_obs_total, n_mtas), dtype=int)
    for m in range(n_mtas):
        mta_obs[m % n_obs_total, m] = 1

    id_data = {
        "modular":     mod_mta,
        "obs_bundles": mta_obs,
        "capacity":    (ctx["capacity"] * elig_scale).astype(int),
    }
    if qid_mta is not None:
        id_data["quadratic"] = qid_mta

    item_data = {"modular": -alpha_1 * np.eye(n_mtas, dtype=np.float64),
                 "weight":  mta_weight}
    if Q_mta is not None:
        item_data["quadratic"] = Q_mta
    input_data = {"id_data": id_data, "item_data": item_data}

    # Covariate-name map: modular | MTA_FE(prices) | quadratic_id | quadratic.
    names = {i: n for i, n in enumerate(mod_names)}
    off = n_id_mod + n_mtas
    for i, n in enumerate(qid_names): names[off + i] = n
    off += n_id_quad
    for i, n in enumerate(quad_names): names[off + i] = n

    n_item_quad = Q_mta.shape[-1] if Q_mta is not None else 0
    meta = dict(
        n_obs=n_obs_total, n_items=n_mtas, n_btas=n_btas, n_mtas=n_mtas,
        n_id_mod=n_id_mod, n_item_mod=n_mtas,
        n_id_quad=n_id_quad, n_item_quad=n_item_quad,
        n_covariates=n_id_mod + n_mtas + n_id_quad + n_item_quad,
        covariate_names=names,
        continental_mta_nums=mta_nums,
    )
    cf = dict(
        A=A, offset_m=offset_m, offset_m_no_xi=offset_m_no_xi,
        beta=beta, gamma_id=gamma_id, gamma_item=gamma_item,
        delta=delta, xi=xi,
        elig=ctx["elig"], pop=ctx["pop"],
    )
    return input_data, meta, cf


def freeze_bounds(config, meta, cf):
    """Pin β, γ_id, γ_item in the combest theta_bounds so the solver only moves
    the MTA item-FEs (prices). Mutates config in place."""
    bounds = config["row_generation"].setdefault("theta_bounds", {})
    lbs = bounds.setdefault("lbs", {})
    ubs = bounds.setdefault("ubs", {})

    # Three parameter blocks to pin. Layout:
    #   [ modular | MTA_FE (unpinned) | quadratic_id | quadratic ]
    fe_end = meta["n_id_mod"] + meta["n_item_mod"]
    blocks = [(0,                            cf["beta"]),
              (fe_end,                       cf["gamma_id"]),
              (fe_end + len(cf["gamma_id"]), cf["gamma_item"])]
    for off, vals in blocks:
        for i, v in enumerate(vals):
            name = meta["covariate_names"][off + i]
            lbs[name] = ubs[name] = float(v)
