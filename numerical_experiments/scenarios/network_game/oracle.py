"""Feature builders and utility evaluator for the network peer-effects game.

Parameter layout (K = 5):
    theta = [ beta_1, beta_2, beta_3, beta_4, delta ]

combest's quadratic subproblem (min-cut reduction) expects the quadratic
feature tensor to be UPPER TRIANGULAR (zeros on and below the diagonal).
With that convention the single quadratic layer we need is triu(D, k=1),
since the game's peer-effect term is

    (delta / 2) * y' D y  =  delta * sum_{s > t} D_{ts} y_t y_s
                          =  delta * y' triu(D, k=1) y,

so the covariate y' Q y (computed by combest as y' (triu(D)) y) is
multiplied by theta_delta to reconstruct the quadratic part of V(y).
"""

from __future__ import annotations

import numpy as np


def build_combest_input(Y, X, D):
    """Build the input_data dict combest expects.

    n_obs = 1, n_items = T, 4 modular covariates + 1 quadratic layer.
    """
    T = X.shape[0]
    assert X.shape == (T, 4), X.shape
    assert D.shape == (T, T), D.shape
    assert Y.shape == (T,), Y.shape

    # (N, T, 4) modular features; N = 1
    modular_agent = X[None, :, :].astype(float)

    # (N, T, T, 1) quadratic features; upper-triangular (combest convention)
    # so that y' Q y = sum_{i<j} D_{ij} y_i y_j and theta_delta scales it to
    # delta * sum_{i<j} D_{ij} y_i y_j = (delta/2) * y' D y.
    quadratic_agent = np.triu(D.astype(float), k=1)[None, :, :, None]

    return {
        "id_data": {
            "obs_bundles": Y[None, :].astype(bool),
            "modular": modular_agent,
            "quadratic": quadratic_agent,
        },
        "item_data": {},
    }


def compute_utility(y, X, D, beta, delta, U):
    """Potential V(y) = (X beta - U)' y + (delta / 2) y' D y."""
    y = y.astype(float)
    L = X @ np.asarray(beta) - U
    return float(L @ y + 0.5 * delta * y @ D @ y)


def naive_probit(Y, X, D):
    """OLS-on-latent probit that ignores simultaneity.

    Fits the single-agent probit y_t = 1{ x_t' beta + delta * (D y)_t > u_t },
    treating (D y)_t as exogenous. Used in the paper as a 'naive' benchmark.
    """
    import warnings
    from scipy.stats import norm
    from scipy.optimize import minimize

    Dy = D.astype(float) @ Y.astype(float)
    Z = np.column_stack([X, Dy])  # (T, 5)
    y = Y.astype(float)

    def negll(theta):
        lin = np.clip(Z @ theta, -30.0, 30.0)
        p = norm.cdf(lin).clip(1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    theta0 = np.zeros(5)
    bounds = [(-5.0, 5.0)] * 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(negll, theta0, method="L-BFGS-B", bounds=bounds)
    return res.x
