import numpy as np


def build_cholesky_factor(error_correlation):
    """Build Cholesky factor from a QUADRATIC registry entry name, or return None."""
    if error_correlation is None:
        return None
    from .registries import QUADRATIC
    from .loaders import load_bta_data, build_context
    raw = load_bta_data()
    ctx = build_context(raw)
    Q = QUADRATIC[error_correlation](ctx)
    Sigma = (Q + Q.T) / 2
    np.fill_diagonal(Sigma, 1.0)
    return np.linalg.cholesky(Sigma)


def build_counterfactual_errors(comm_manager, n_btas, A, offset, seed,
                                elig=None, error_scaling=None,
                                L_corr=None):
    """Build local modular errors for counterfactual MTA-level estimation.

    Returns (local_errors, oracle_fn, oracle_takes_data).
    """
    n_items = A.shape[0]
    local_errors = np.zeros((comm_manager.num_local_agent, n_items))
    for i, gid in enumerate(comm_manager.agent_ids):
        obs_id = comm_manager.obs_ids[i]
        rng = np.random.default_rng((seed, gid))
        bta_err = rng.normal(0, 1, n_btas)
        if L_corr is not None:
            bta_err = L_corr @ bta_err
        if error_scaling == "elig" and elig is not None:
            bta_err *= elig[obs_id]
        local_errors[i] = bta_err @ A.T + offset
    return local_errors
