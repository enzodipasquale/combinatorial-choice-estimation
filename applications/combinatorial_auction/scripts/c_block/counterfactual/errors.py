"""Counterfactual error construction for MTA-level estimation."""
import numpy as np


def build_counterfactual_errors(comm_manager, n_btas, A, offset, seed,
                                elig=None, error_scaling=None,
                                L_corr=None, pop=None):
    """Build local modular errors for counterfactual MTA-level estimation."""
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
        elif error_scaling == "pop" and pop is not None:
            bta_err *= pop
        local_errors[i] = bta_err @ A.T + offset
    return local_errors
