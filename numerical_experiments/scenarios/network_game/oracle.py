"""combest feature encoder for the network peer-effects game.

Parameter layout (K = 5):
    theta = [ beta_1, beta_2, beta_3, beta_4, delta ]

combest's quadratic subproblem (min-cut reduction) expects the quadratic
feature tensor to be UPPER TRIANGULAR (zeros on and below the diagonal).
The single quadratic layer we need is therefore triu(D, k=1), since the
game's peer-effect term is

    (delta / 2) * y' D y  =  delta * sum_{s > t} D_{ts} y_t y_s
                          =  delta * y' triu(D, k=1) y.

Passing a symmetric Q here (both triangles filled) silently double-counts
each pair in combest's posiform reduction and yields the wrong argmax.
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

    modular_agent = X[None, :, :].astype(float)                       # (1, T, 4)
    quadratic_agent = np.triu(D.astype(float), k=1)[None, :, :, None] # (1, T, T, 1)

    return {
        "id_data": {
            "obs_bundles": Y[None, :].astype(bool),
            "modular": modular_agent,
            "quadratic": quadratic_agent,
        },
        "item_data": {},
    }


def naive_probit(Y, X, D):
    """Naive single-agent probit on the observed Y (ignores simultaneity).

    Fits y_t = 1{x_t' beta + delta * (D Y)_t > u_t} treating (D Y)_t as
    exogenous. Useful as a researcher-would-actually-use-it starting point
    for the SML optimizer / importance-sampler anchor (paper's Figure 2
    also plots naive-probit estimates as a reference).

    Returns theta_hat of length 5: (beta_1, ..., beta_4, delta).
    """
    import warnings
    from scipy.optimize import minimize
    from scipy.stats import norm

    Dy = D.astype(float) @ Y.astype(float)
    Z = np.column_stack([X, Dy])              # (T, 5)
    y = Y.astype(float)

    def negll(theta):
        lin = np.clip(Z @ theta, -30.0, 30.0)
        p = norm.cdf(lin).clip(1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    bounds = [(-5.0, 5.0)] * 4 + [(0.0, 5.0)]  # matches SML's delta >= 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(negll, np.zeros(5), method="L-BFGS-B", bounds=bounds)
    return res.x
