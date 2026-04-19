"""Modular-error oracle construction.

Two responsibilities, split so the MPI caller can compute artifacts once on
rank 0 and broadcast them before every rank calls `install` (BTA-level) or
`install_aggregated` (counterfactual: BTA-drawn, MTA-aggregated).

Config keys consumed (from the `application` block):
    error_seed          (int, required)
    error_correlation   (str | null)   Name of a QUADRATIC covariate whose
                                        normalized form drives a Cholesky
                                        correlation on the BTA error vector.
    spatial_rho         (float | null) Σ = (I−ρW)⁻¹(I−ρW)⁻ᵀ over the
                                        symmetric, binarized, row-normalized
                                        BTA adjacency.  Rescaled to unit diag.
    error_scaling       ('pop'|'elig'|null)   In-place post-draw scaling.
    sigma_price         (float, optional)     Activates a correlated price leg:
                                              ε_ij = pop_j·N¹ + price_j·N².
    rho_pop_price       (float in (-1,1), optional; only used if sigma_price>0)

Mutually exclusive combinations:
  • `error_correlation` and `spatial_rho` (only one covariance source).
  • `sigma_price > 0` incompatible with any `cov` (price leg assumes iid).
"""
import numpy as np

from ..data.loaders import cholesky_factor


# ── Covariance builders ───────────────────────────────────────────────

def build_sar_correlation(bta_adjacency, rho):
    """Σ = (I − ρW)⁻¹(I − ρW)⁻ᵀ rescaled to unit diagonal, where W is the
    symmetric, binarized, row-normalized adjacency. Raises on |ρ|≥1 or non-PSD.
    """
    if abs(rho) >= 1:
        raise ValueError(f"spatial_rho must satisfy |ρ|<1, got {rho}")

    adj = ((bta_adjacency + bta_adjacency.T) > 0).astype(float)
    np.fill_diagonal(adj, 0.0)
    row = adj.sum(1, keepdims=True)
    W = np.where(row > 0, adj / row, 0.0)

    M_inv = np.linalg.solve(np.eye(adj.shape[0]) - rho * W, np.eye(adj.shape[0]))
    S = M_inv @ M_inv.T
    d = np.sqrt(np.diag(S))
    S /= np.outer(d, d)
    np.fill_diagonal(S, 1.0)

    np.linalg.cholesky(S)   # PSD check
    return S


def covariance(ctx, app):
    """Return the BTA covariance Σ or None. Consumes error_correlation /
    spatial_rho from `app` (mutually exclusive)."""
    correlation = app.get("error_correlation")
    spatial_rho = app.get("spatial_rho")
    if correlation is not None and spatial_rho is not None:
        raise ValueError("error_correlation and spatial_rho are mutually exclusive")
    if spatial_rho is not None:
        return build_sar_correlation(ctx["bta_adjacency"], float(spatial_rho))
    if correlation is not None:
        L = cholesky_factor(ctx, correlation)
        return L @ L.T
    return None


# ── BTA-level oracle (estimation) ─────────────────────────────────────

def _install_pop_with_price_leg(model, *, seed, pop, price, sigma_price, rho):
    """ε_ij = pop_j·N¹_ij + price_j·N²_ij with corr(N¹,N²)=ρ at each (i,j),
    iid across (i,j). Activated when `scaling='pop' and sigma_price>0`."""
    if pop is None or price is None:
        raise ValueError("sigma_price>0 requires both pop and price vectors")
    s, r = float(sigma_price), float(rho or 0.0)
    if s < 0:
        raise ValueError(f"sigma_price must be ≥0, got {s}")
    if not -1 < r < 1:
        raise ValueError(f"rho_pop_price must satisfy |ρ|<1, got {r}")

    cm = model.features.comm_manager
    n_items = pop.shape[0]
    errors = np.zeros((cm.num_local_agent, n_items))
    c = np.sqrt(1 - r * r)
    for i, gid in enumerate(cm.agent_ids):
        z = np.random.default_rng((seed, gid)).standard_normal((2, n_items))
        errors[i] = pop * z[0] + price * (s * (r * z[0] + c * z[1]))
    model.features.local_modular_errors = errors
    model.features.set_error_oracle(
        lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    )


def install(model, *, seed, cov, scaling=None, pop=None, price=None,
            sigma_price=None, rho=None):
    """Install the modular-error oracle on every rank.

    • scaling='pop' + sigma_price>0 → two-field custom oracle (iid by assumption).
    • otherwise → combest's Cholesky-correlated oracle with optional post-scale
      by pop or elig.
    """
    s = float(sigma_price or 0.0)
    if scaling == "pop" and s > 0:
        if cov is not None:
            raise ValueError(
                "scaling='pop' with sigma_price>0 is incompatible with "
                "error_correlation / spatial_rho (price leg assumes iid)"
            )
        _install_pop_with_price_leg(model, seed=seed, pop=pop, price=price,
                                     sigma_price=s, rho=rho)
        return

    model.features.build_local_modular_error_oracle(seed=seed, covariance_matrix=cov)
    if scaling == "pop":
        if pop is None:
            raise ValueError("scaling='pop' requires pop vector")
        model.features.local_modular_errors *= pop[None, :]
    elif scaling == "elig":
        elig = model.data.local_data.id_data["elig"]   # local per rank
        model.features.local_modular_errors *= elig[:, None]
    elif scaling is not None:
        raise ValueError(f"error_scaling must be 'pop', 'elig' or null, got {scaling!r}")


# ── MTA-level oracle (counterfactual) ─────────────────────────────────

def install_aggregated(model, *, seed, A, bta_cov=None, offset=None,
                       scaling=None, pop=None, elig=None):
    """Draw BTA-level modular errors, optionally Cholesky-correlate and scale
    them, aggregate to MTA level via A, and add a deterministic MTA offset.

    Called by the counterfactual: items are MTAs but stochasticity is BTA-level.
    Each rank fills its own local_modular_errors from its local agent list.
    """
    cm = model.features.comm_manager
    n_bta, n_items = A.shape[1], A.shape[0]
    L = np.linalg.cholesky(bta_cov) if bta_cov is not None else None

    errors = np.zeros((cm.num_local_agent, n_items))
    for i, gid in enumerate(cm.agent_ids):
        e = np.random.default_rng((seed, gid)).normal(0, 1, n_bta)
        if L is not None:
            e = L @ e
        if scaling == "elig":
            e *= elig[cm.obs_ids[i]]
        elif scaling == "pop":
            e *= pop
        errors[i] = e @ A.T + (offset if offset is not None else 0.0)

    model.features.local_modular_errors = errors
    model.features.set_error_oracle(
        lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    )
