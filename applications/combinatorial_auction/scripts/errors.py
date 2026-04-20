"""Modular-error oracle construction.

Two responsibilities, split so the MPI caller can compute artifacts once on
rank 0 and broadcast them before every rank calls `install` (BTA-level) or
`install_aggregated` (counterfactual: BTA-drawn, MTA-aggregated).

Config keys consumed from the `application` block:

    error_seed          int      Required. Base seed for per-agent draws.
    error_scaling       str|null 'pop' or 'elig' (or absent / null for iid).

BTA error draws are iid N(0,1), optionally post-scaled by pop or elig.
"""
import numpy as np


# Kept as a stub for compatibility with estimate.py / run.py — both call
# ``covariance(ctx, app)`` but we no longer support non-trivial covariance.
def covariance(ctx, app):
    return None


def install(model, *, seed, cov=None, scaling=None, pop=None, price=None,
            sigma_price=None, rho=None):
    """BTA-level modular-error oracle. `cov`, `price`, `sigma_price`, `rho`
    are accepted (and ignored) for call-site compatibility."""
    model.features.build_local_modular_error_oracle(seed=seed, covariance_matrix=None)
    if scaling == "pop":
        if pop is None:
            raise ValueError("scaling='pop' requires pop vector")
        model.features.local_modular_errors *= pop[None, :]
    elif scaling == "elig":
        elig = model.data.local_data.id_data["elig"]
        model.features.local_modular_errors *= elig[:, None]
    elif scaling is not None:
        raise ValueError(f"error_scaling must be 'pop', 'elig' or null, got {scaling!r}")


def install_aggregated(model, *, seed, A, offset=None,
                       scaling=None, pop=None, elig=None, bta_cov=None):
    """Draw BTA-level modular errors, scale by pop / elig, aggregate to MTA
    via A, add a deterministic MTA offset.  Used by the counterfactual.
    (`bta_cov` accepted and ignored for call-site compatibility.)
    """
    cm = model.features.comm_manager
    n_bta = A.shape[1]
    n_items = A.shape[0]

    errors = np.zeros((cm.num_local_agent, n_items))
    for i, gid in enumerate(cm.agent_ids):
        e = np.random.default_rng((seed, gid)).normal(0, 1, n_bta)
        if scaling == "elig":
            e *= elig[cm.obs_ids[i]]
        elif scaling == "pop":
            e *= pop
        errors[i] = e @ A.T + (offset if offset is not None else 0.0)

    model.features.local_modular_errors = errors
    model.features.set_error_oracle(
        lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    )
