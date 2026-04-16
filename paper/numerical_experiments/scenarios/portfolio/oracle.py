"""Covariates and error oracles for the portfolio-choice scenario."""

import numpy as np


def covariates_oracle(bundles, ids, data):
    """Compute features phi(s, w) for given bundles.

    Parameters
    ----------
    bundles : ndarray (n, 2*M)
        First M columns are s (inclusion), last M columns are w (weights).
    ids : array-like
        Local agent indices into data.id_data arrays.
    data : LocalData
        Contains id_data['X_agents'] (n_local, M, K) and
        item_data['Sigma'] (M, M).

    Returns
    -------
    features : ndarray (n, 5)
        Columns correspond to theta = [beta_0, beta_1, beta_2, gamma, kappa].
    """
    X_agents = data.id_data['X_agents']  # (n_local, M, K)
    Sigma = data.item_data['Sigma']       # (M, M)
    M = Sigma.shape[0]

    s = bundles[:, :M]
    w = bundles[:, M:]

    n = bundles.shape[0]
    K = X_agents.shape[2]
    features = np.zeros((n, K + 2))

    for i_local, idx in enumerate(ids):
        X_i = X_agents[idx]  # (M, K)
        w_i = w[i_local]     # (M,)
        s_i = s[i_local]     # (M,)

        # phi_beta_k = sum_j w_j * X_ij_k
        features[i_local, :K] = w_i @ X_i  # (K,)

        # phi_gamma = -0.5 * w^T Sigma w
        features[i_local, K] = -0.5 * w_i @ Sigma @ w_i

        # phi_kappa = -sum_j s_j
        features[i_local, K + 1] = -s_i.sum()

    return features


def error_oracle(bundles, ids, data):
    """Zero-noise error oracle. Returns zeros."""
    return np.zeros(len(ids))
