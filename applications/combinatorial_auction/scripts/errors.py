"""Unified error-oracle construction for the combinatorial auction.

Two responsibilities, split so the MPI caller can compute artifacts once on
rank 0 and broadcast them before every rank calls `install`.

Config keys consumed (from the `application` block):
    error_seed          (int, required)
    error_correlation   (str | null)    Name of a QUADRATIC covariate whose
                                         normalized version is used as a
                                         correlation matrix (Cholesky).
    spatial_rho         (float | null)  Σ = (I−ρW)⁻¹(I−ρW)⁻ᵀ over symmetrized,
                                         binarized, row-normalized BTA adjacency.
    error_scaling       ('pop'|'elig'|null)  In-place post-draw scaling.

error_correlation and spatial_rho are mutually exclusive.
"""
import numpy as np

from ..data.loaders import cholesky_factor


def build_sar_correlation(bta_adjacency, rho):
    """Σ = (I − ρW)⁻¹(I − ρW)⁻ᵀ rescaled to unit diagonal, where W is the
    symmetric, binarized, row-normalized adjacency. Raises on |ρ|≥1 or non-PSD."""
    if abs(rho) >= 1:
        raise ValueError(f"spatial_rho must satisfy |ρ| < 1, got {rho}")

    adj = ((bta_adjacency + bta_adjacency.T) > 0).astype(float)
    np.fill_diagonal(adj, 0.0)
    row = adj.sum(1, keepdims=True)
    W = np.where(row > 0, adj / row, 0.0)

    M_inv = np.linalg.solve(np.eye(adj.shape[0]) - rho * W, np.eye(adj.shape[0]))
    S = M_inv @ M_inv.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    np.fill_diagonal(S, 1.0)

    np.linalg.cholesky(S)  # PSD check
    return S


def covariance(ctx, app):
    """Return the error covariance Σ (n_items, n_items) or None for iid.

    Call once on rank 0 and broadcast. Consumes error_correlation / spatial_rho.
    """
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


def pop_vector(ctx):
    """Return the normalized per-BTA pop vector used for error_scaling='pop'.

    This is the same ctx['pop'] that appears in the elig_pop covariate and in
    any pop-centroid quadratic — guaranteed consistent because it's a single
    reference into the shared context.
    """
    return ctx["pop"]


def install_aggregated(model, *, seed, A, bta_cov=None, offset=None,
                       scaling=None, pop=None, elig=None):
    """Install a custom oracle that draws BTA-level modular errors, optionally
    Cholesky-correlates and scales them, aggregates them to MTA level via A,
    and adds a deterministic MTA-level offset.

    Used by the counterfactual: the model's items are MTAs but the underlying
    stochasticity is at BTA level. Each rank fills its local_modular_errors
    from its own local agent list. Must be called on every rank.
    """
    cm = model.features.comm_manager
    n_bta, n_items = A.shape[1], A.shape[0]
    L = np.linalg.cholesky(bta_cov) if bta_cov is not None else None

    errors = np.zeros((cm.num_local_agent, n_items))
    for i, gid in enumerate(cm.agent_ids):
        rng = np.random.default_rng((seed, gid))
        e = rng.normal(0, 1, n_bta)
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


def install(model, *, seed, cov, scaling=None, pop=None):
    """Install modular error oracle into `model.features`, on every rank.

    `cov` and `pop` are computed on rank 0 and broadcast by the caller.
    """
    model.features.build_local_modular_error_oracle(seed=seed, covariance_matrix=cov)
    if scaling == "pop":
        if pop is None:
            raise ValueError("scaling='pop' requires pop vector")
        model.features.local_modular_errors *= pop[None, :]
    elif scaling == "elig":
        # Per-rank: combest distributes id_data so each rank's elig is local.
        elig = model.data.local_data.id_data["elig"]
        model.features.local_modular_errors *= elig[:, None]
    elif scaling is not None:
        raise ValueError(f"error_scaling must be 'pop', 'elig' or null, got {scaling!r}")
