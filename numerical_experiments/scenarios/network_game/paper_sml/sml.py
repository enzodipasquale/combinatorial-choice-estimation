"""Simulated MLE for the peer-effects game via the paper's scenario sampler
(Graham and Gonzalez 2023), with scenario recycling (Appendix D).

For a single observed game (X, D, Y), Pr(Y | X; θ) = sum_{b in B_Y} ζ(b; θ)
with B_Y the set of scenarios (T-dim rectangles in R^T) whose minimal NE
equals Y. Paper's importance-sampled estimate is
   Pr̂(Y | X; θ) = (1/S) sum_s ζ(B̃^{(s)}; θ) / λ_y(B̃^{(s)}; θ^{(0)}).

Scenario recycling: at sampling time (at θ^{(0)}) we pin, per agent, the
delta-coefficient of the bucket boundary. With a SINGLE strategic parameter
δ, these boundaries are affine in θ, so re-evaluating ζ at any θ costs
O(S * T) — no Threshold Finder re-runs. See `bucket_bounds` below.

The file also provides helpers to reproduce paper's Table 1 Panel B:

  * ``fit_sml_constrained_delta`` — MLE with δ fixed (for the LR test).
  * ``numerical_hessian`` — symmetric central-difference Hessian of
    ``simulated_loglik`` at θ̂, used for the Wald-SE and CI reports.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .scenario_sampler import sample_scenario, sample_scenario_at_theta


# ---------------------------------------------------------------------------
# Bucket bounds as affine functions of θ
# ---------------------------------------------------------------------------

def bucket_bounds(X, beta, delta, coef_lo, coef_hi):
    """Affine sub-bucket bounds under scenario recycling (paper Appendix D).

    Given the per-agent integer delta-multipliers ``coef_lo`` and ``coef_hi``
    cached at sampling time (see ``scenario_sampler._sample_scenario_min``),
    the scenario's bucket boundaries are affine in theta:

        b̌_t(theta) = X_t' beta + delta * coef_lo[t]   (−∞ sentinel allowed)
        b̄_t(theta) = X_t' beta + delta * coef_hi[t]   (+∞ sentinel allowed)

    Returns (lowers, uppers), both shape (T,).
    """
    Xb = X @ beta
    lowers = np.where(np.isfinite(coef_lo), Xb + delta * coef_lo, -np.inf)
    uppers = np.where(np.isfinite(coef_hi), Xb + delta * coef_hi,  np.inf)
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
    """Draw S scenarios at θ^{(0)} = (beta0, delta0). Each scenario stores:
      * ``coef_lo``, ``coef_hi``: per-agent integer δ-multipliers for the
        sub-bucket boundaries (±inf sentinels where a side is unbounded).
      * ``log_mass0``: Σ_t log[F(b̄_t^(0)) − F(b̌_t^(0))] evaluated at θ^{(0)}.
      * ``log_omega``: Σ_t log ω_t(θ^{(0)}) (paper Eq 11), the constant
        denominator of the importance-weight ratio.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(S):
        _U, c_lo, c_hi, lm0, lw = sample_scenario(
            X, D, Y, beta0, delta0, rng, selection=selection)
        scenarios.append({"coef_lo":   c_lo,
                          "coef_hi":   c_hi,
                          "log_mass0": lm0,
                          "log_omega": lw})
    return scenarios


def simulated_loglik(X, D, Y, beta, delta, scenarios, selection="min"):
    """Evaluate log Pr̂(Y|X;θ) = log[(1/S) Σ_s ζ(B̃_s;θ)/λ_y(B̃_s;θ^{(0)})].

    Implements paper's Eq (3) with scenario recycling (Appendix D): each
    scenario's sub-bucket boundaries are affine in theta via ``coef_lo``
    and ``coef_hi``; ``log_omega`` and ``log_mass0`` are the constants
    making up log λ_y(B̃_s; θ^{(0)}) = log_omega + log_mass0.

    log-ratio for scenario s:
        log ζ(B̃_s; θ) − log λ_y(B̃_s; θ^{(0)})
      = Σ_t log[F(b̄_t(θ)) − F(b̌_t(θ))]   −   log_omega_s   −   log_mass0_s.
    """
    if selection != "min":
        raise NotImplementedError("Only 'min' selection supported.")
    S = len(scenarios)
    if S == 0:
        return -np.inf
    log_ratios = np.empty(S)
    for s, sc in enumerate(scenarios):
        lo, up = bucket_bounds(X, beta, delta, sc["coef_lo"], sc["coef_hi"])
        log_zeta = log_bucket_mass(lo, up).sum()
        log_ratios[s] = log_zeta - sc["log_omega"] - sc["log_mass0"]
    m = log_ratios.max()
    return float(m + np.log(np.exp(log_ratios - m).mean()))


# ---------------------------------------------------------------------------
# CRN simulated log-likelihood (paper Appendix D first paragraph)
# ---------------------------------------------------------------------------

def simulated_loglik_crn(X, D, Y, beta, delta, uniforms_S):
    """CRN-protocol simulated log-likelihood (paper Appendix D, first
    paragraph).

    Given pre-drawn S × T standard uniforms held fixed across optimization,
    regenerate scenarios at the current (beta, delta) via inverse-CDF on
    the truncated normal densities of Algorithm 3. The IS proposal is at
    the current θ (not at a fixed anchor θ⁽⁰⁾), so the per-scenario IS
    ratio collapses to ζ/λ = 1/Πₜ ωₜ(θ).

        log L̂(θ) = log[(1/S) Σ_s exp(-log_omega^{(s)}(θ))].
    """
    S = uniforms_S.shape[0]
    log_ratios = np.empty(S)
    for s in range(S):
        log_omega = sample_scenario_at_theta(
            X, D, Y, beta, delta, uniforms_S[s])
        log_ratios[s] = -log_omega
    m = log_ratios.max()
    return float(m + np.log(np.exp(log_ratios - m).mean()))


def draw_uniforms(S, T, seed):
    """S × T standard uniforms, fixed across optimization (CRN protocol)."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(S, T))


def fit_sml_crn(X, D, Y, uniforms_S, theta_init=None, verbose=False,
                bounds=None, max_iter=200):
    """SML fit using the CRN protocol (paper Appendix D first paragraph).

    Pre-drawn S × T uniforms `uniforms_S` are held fixed; scenarios are
    regenerated at every θ during L-BFGS-B via inverse-CDF. No anchor
    θ⁽⁰⁾, no recycling. Returns (theta_hat, scipy_result).
    """
    if theta_init is None:
        theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    if bounds is None:
        bounds = [(-5.0, 5.0)] * 4 + [(0.0, 5.0)]

    def negll(theta):
        beta = theta[:4]
        delta = float(max(theta[4], 0.0))
        return -simulated_loglik_crn(X, D, Y, beta, delta, uniforms_S)

    res = minimize(negll, theta_init, method="L-BFGS-B", bounds=bounds,
                   options={"disp": verbose, "maxiter": max_iter})
    return res.x, res


def fit_sml_crn_constrained_delta(X, D, Y, uniforms_S, delta_fixed,
                                   beta_init=None, verbose=False,
                                   max_iter=200):
    """CRN-protocol SML with δ fixed at `delta_fixed` (for LR test).

    Reuses the same `uniforms_S` as the unconstrained fit so the LR
    statistic is computed on the same simulated likelihood.
    """
    if beta_init is None:
        beta_init = np.zeros(4)

    def negll(beta):
        return -simulated_loglik_crn(X, D, Y, beta, delta_fixed, uniforms_S)

    res = minimize(negll, beta_init, method="L-BFGS-B",
                   bounds=[(-5.0, 5.0)] * 4,
                   options={"disp": verbose, "maxiter": max_iter})
    return res.x, res


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
