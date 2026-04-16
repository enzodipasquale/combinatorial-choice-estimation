"""Portfolio-choice DGP: generate items, covariance, and true parameters."""

import numpy as np


def generate_dgp(seed=42, N=200, M=15, K=3, gamma=None, kappa=None,
                  eta_std=None, cov_mode=None):
    """Build the synthetic portfolio-choice environment.

    For M <= 20 (baseline): factor-model covariance, low agent noise.
    For M > 20 (scaled-up): equicorrelated covariance, higher risk
    aversion and agent noise to maintain portfolio diversity.

    Returns
    -------
    X : ndarray (M, K)
        Base characteristic matrix.
    X_agents : ndarray (N, M, K)
        Per-agent characteristic matrices (X + agent-specific noise).
    Sigma : ndarray (M, M)
        Asset covariance matrix (symmetric positive definite).
    theta_star : ndarray (5,)
        True parameters [beta_0, beta_1, beta_2, gamma, kappa].
    """
    large = M > 20

    if gamma is None:
        gamma = 2.0 if not large else 20.0
    if kappa is None:
        kappa = 0.05 if not large else 0.0005
    if eta_std is None:
        eta_std = 0.1 if not large else 0.3
    if cov_mode is None:
        cov_mode = 'factor' if not large else 'diagonal'

    rng = np.random.default_rng(seed)

    # Characteristic matrix
    X = rng.standard_normal((M, K))

    # Per-agent shifters
    eta = rng.normal(0, eta_std, size=(N, M, K))
    X_agents = X[np.newaxis, :, :] + eta  # (N, M, K)

    # Covariance matrix
    if cov_mode == 'factor':
        n_factors = 2
        Lambda = rng.standard_normal((M, n_factors))
        sigma_xi = 0.3
        Sigma = Lambda @ Lambda.T + sigma_xi**2 * np.eye(M)
    elif cov_mode == 'equicorr':
        rho = 0.5
        Sigma = (1 - rho) * np.eye(M) + rho * np.ones((M, M))
    elif cov_mode == 'diagonal':
        Sigma = np.eye(M)
    else:
        raise ValueError(f"Unknown cov_mode: {cov_mode}")

    # True parameters
    theta_star = np.array([1.0, 0.5, -0.3, gamma, kappa])

    return X, X_agents, Sigma, theta_star
