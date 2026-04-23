"""Simulated MLE for the peer-effects game via the scenario sampler.

For a single observed game (X, D, Y), the likelihood is
   Pr(Y | X; θ) = sum_{b in B_Y} ζ(b; θ)
with ζ(b; θ) = prod_t [F(b_upper_t; θ) - F(b_lower_t; θ)] and B_Y the set
of scenarios (T-dim rectangles in ℝ^T) whose minimal NE equals Y.

Paper's SML estimator draws S scenarios from an importance distribution
λ_y(b; θ^{(0)}) and estimates
   Pr̂(Y | X; θ) = (1/S) sum_s ζ(B̃^{(s)}; θ) / λ_y(B̃^{(s)}; θ^{(0)}).

With a pre-drawn set of S scenarios at θ^{(0)}, the bucket boundaries
(b_lower, b_upper) are affine functions of θ (through X_t'β and δ·(D Y)_t —
see Section 3.1). So we can re-evaluate ζ at any θ without re-sampling, as
long as the scenario rectangles themselves don't change (scenario recycling,
Appendix D).

IMPORTANT: For the peer-effects game with a SINGLE strategic parameter
δ, the paper proves scenario recycling is exact (buckets are continuous
affine functions of θ). We exploit this.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .scenario_sampler import sample_scenario


# ---------------------------------------------------------------------------
# Bucket bounds as affine functions of θ
# ---------------------------------------------------------------------------

def bucket_bounds(X, D, Y, beta, delta, U, selection="min"):
    """Given U drawn at (θ^{(0)}), evaluate bucket boundaries at θ = (β, δ).

    For minimal NE with peer effects:
      - If y_t = 0: bucket is (x_t'β + δ·(DY)_t , ∞)   → lower = that, upper = ∞
      - If y_t = 1: bucket is (−∞, h_t(θ; U_<=t)]
        where h_t was pinned at sampling time. Under scenario recycling,
        h_t is re-computed at the new θ via the Threshold Finder on U (which
        is held fixed).
    """
    # Importing here to avoid a circular import with scenario_sampler.
    from .scenario_sampler import _threshold_finder

    if selection != "min":
        raise NotImplementedError("Only 'min' selection supported.")

    T = X.shape[0]
    Xb = X @ beta
    lowers = np.full(T, -np.inf)
    uppers = np.full(T, np.inf)
    for t in range(T):
        if not Y[t]:
            lowers[t] = Xb[t] + delta * (D[t].astype(float) @ Y.astype(float))
            uppers[t] = np.inf
    for t in range(T):
        if Y[t]:
            h = _threshold_finder(X, D, Y, beta, delta, U, t)
            lowers[t] = -np.inf
            uppers[t] = h

    return lowers, uppers


def log_bucket_mass(lowers, uppers):
    out = np.empty_like(lowers, dtype=float)
    for i in range(len(lowers)):
        lo, up = lowers[i], uppers[i]
        if lo == -np.inf and up == np.inf:
            out[i] = 0.0
        elif lo == -np.inf:
            out[i] = norm.logcdf(up)
        elif up == np.inf:
            out[i] = norm.logsf(lo)
        else:
            out[i] = np.log(max(norm.cdf(up) - norm.cdf(lo), 1e-300))
    return out


# ---------------------------------------------------------------------------
# Simulated log-likelihood
# ---------------------------------------------------------------------------

def draw_scenarios(X, D, Y, beta0, delta0, S, seed, selection="min"):
    """Draw S scenarios at θ^{(0)} = (beta0, delta0). Returns list of dicts."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for s in range(S):
        U, lo, up, lm0 = sample_scenario(
            X, D, Y, beta0, delta0, rng, selection=selection)
        scenarios.append({"U": U, "lower0": lo, "upper0": up,
                          "log_mass0": lm0})
    return scenarios


def simulated_loglik(X, D, Y, beta, delta, scenarios, selection="min"):
    """Evaluate log Pr̂(Y|X; θ) using S pre-drawn scenarios."""
    S = len(scenarios)
    if S == 0:
        return -np.inf
    # log[ (1/S) sum_s exp(log_mass_theta(s) - log_mass_0(s)) ]
    log_ratios = np.empty(S)
    for s, sc in enumerate(scenarios):
        lo, up = bucket_bounds(X, D, Y, beta, delta, sc["U"], selection)
        log_mass = log_bucket_mass(lo, up).sum()
        log_ratios[s] = log_mass - sc["log_mass0"]
    # logsumexp
    m = log_ratios.max()
    return float(m + np.log(np.exp(log_ratios - m).mean()))


def fit_sml(X, D, Y, S, seed, selection="min", theta_init=None,
            beta0=None, delta0=None, verbose=False):
    """Run one SML estimate.

    theta_init is the starting value for the optimizer (length 5:
    [β_1, β_2, β_3, β_4, δ]). If None we use a reasonable default.

    beta0, delta0 are the importance-sampler anchor θ^{(0)}. If None we
    use theta_init.
    """
    if theta_init is None:
        theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    if beta0 is None:
        beta0 = theta_init[:4]
    if delta0 is None:
        delta0 = float(theta_init[4])

    scenarios = draw_scenarios(X, D, Y, beta0, delta0, S, seed, selection)

    def negll(theta):
        beta = theta[:4]
        delta = float(max(theta[4], 0.0))   # delta >= 0
        return -simulated_loglik(X, D, Y, beta, delta, scenarios, selection)

    bounds = [(-5.0, 5.0)] * 4 + [(0.0, 5.0)]
    res = minimize(negll, theta_init, method="L-BFGS-B", bounds=bounds,
                   options={"disp": verbose, "maxiter": 200})
    return res.x, res, scenarios


def fit_sml_constrained_delta(X, D, Y, delta_fixed, S, seed,
                              selection="min", beta_init=None,
                              beta0=None, delta0=None, verbose=False):
    """Run SML with delta fixed at `delta_fixed` (for LR test).

    Returns (beta_hat, result_object, scenarios).  Optimizes over the 4-dim
    β subvector only.  The importance-sampler anchor (beta0, delta0) is
    still drawn at whatever (beta0, delta0) is passed in; delta0 need not
    equal delta_fixed (scenario recycling is valid at any theta under the
    peer-effects-game assumption).
    """
    if beta_init is None:
        beta_init = np.zeros(4)
    if beta0 is None:
        beta0 = beta_init
    if delta0 is None:
        delta0 = delta_fixed

    scenarios = draw_scenarios(X, D, Y, beta0, delta0, S, seed, selection)

    def negll(beta):
        return -simulated_loglik(X, D, Y, beta, delta_fixed, scenarios,
                                 selection)

    res = minimize(negll, beta_init, method="L-BFGS-B",
                   bounds=[(-5.0, 5.0)] * 4,
                   options={"disp": verbose, "maxiter": 200})
    return res.x, res, scenarios


# ---------------------------------------------------------------------------
# Numerical Hessian of the simulated log-likelihood
# ---------------------------------------------------------------------------

def numerical_hessian(f, theta, h=1e-4):
    """Symmetrized central-difference Hessian of a scalar f(theta).

    Uses standard O(h^2) finite differences:
      * Diagonal:   [f(x+h*e_i) - 2*f(x) + f(x-h*e_i)] / h^2
      * Off-diag:   [f(x+h*e_i+h*e_j) - f(x+h*e_i-h*e_j)
                    -f(x-h*e_i+h*e_j) + f(x-h*e_i-h*e_j)] / (4 h^2)

    51 function evaluations for n=5. Cheap under scenario recycling
    because each simulated_loglik call reuses the same scenarios.
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    H = np.zeros((n, n))
    f0 = f(theta)
    # Precompute single-axis shifts (used for diagonals and reused below).
    f_plus  = np.empty(n)
    f_minus = np.empty(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        f_plus[i]  = f(theta + ei)
        f_minus[i] = f(theta - ei)
        H[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (h * h)
    # Off-diagonals (upper triangle; symmetrize).
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            fpp = f(theta + ei + ej)
            fpm = f(theta + ei - ej)
            fmp = f(theta - ei + ej)
            fmm = f(theta - ei - ej)
            val = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            H[i, j] = H[j, i] = val
    return H
