"""Scenario Sampler for the peer-effects game (Graham and Gonzalez 2023).

Implements the minimal-NE variant (paper Algorithms 1 and 2, peer-effects
case). Shock distribution: U_t ~ N(0, 1) iid (Appendix B).

`sample_scenario` returns the tuple (U, lowers, uppers, log_mass,
delta_coef), where delta_coef is the per-agent coefficient needed to
re-evaluate the bucket boundaries affinely in theta via the paper's
scenario-recycling trick (Appendix D). See `sml.bucket_bounds`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Minimal NE finder (used inside the threshold finder)
# ---------------------------------------------------------------------------

def _minimal_ne(X, D, U, beta, delta):
    T = X.shape[0]
    Xb = X @ beta
    y = np.zeros(T, dtype=bool)
    for _ in range(T + 1):
        margin = Xb + delta * (D @ y.astype(float)) - U
        y_new = margin >= 0
        if np.array_equal(y_new, y):
            return y
        y = y_new
    return y


# ---------------------------------------------------------------------------
# Threshold Finder (Algorithm 2, peer-effects case)
# ---------------------------------------------------------------------------

def _threshold_finder(X, D, Y, beta, delta, U, t):
    """Compute threshold h_t and return the pinned minimal NE Ytilde.

    At sampling time we invoke this with (beta_0, delta_0) and cache the
    *pinned* Ytilde for later scenario recycling. For a new theta under
    recycling we do NOT re-invoke the minimal-NE solver: the bucket
    boundary h_t(theta) = X_t' beta + delta * D_t @ Ytilde is affine in
    theta with Ytilde held fixed.

    Returns:
        h_t    (float): threshold at the sampling theta (beta, delta).
        Ytilde (bool array, shape (T,)): the pinned minimal NE used to
               compute h_t. Cache this so bucket bounds can be re-evaluated
               affinely at any new theta.
    """
    T = X.shape[0]
    Xb = X @ beta
    Utilde = np.empty(T)
    for tp in range(T):
        if not Y[tp]:
            Utilde[tp] = U[tp]                   # y_{t'} = 0 → already drawn
        else:
            if tp < t:
                Utilde[tp] = U[tp]               # h_{t'} already found
            else:
                Utilde[tp] = Xb[tp] - 1.0        # provisional low shock
    Utilde[t] = Xb[t] + delta * (D[t].astype(float) @ Y.astype(float)) + 1.0
    Ytilde = _minimal_ne(X, D, Utilde, beta, delta)
    h_t = Xb[t] + delta * (D[t].astype(float) @ Ytilde.astype(float))
    return h_t, Ytilde


# ---------------------------------------------------------------------------
# Scenario Sampler (Algorithm 1, peer-effects case)
# ---------------------------------------------------------------------------

def _sample_scenario_min(X, D, Y, beta, delta, rng):
    """Sample U and identify the scenario b ∈ B_y in which U lands.

    Implements paper's Algorithm 1 (peer-effects case) in full, including
    Step 3.b (scenario identification). Each agent's bucket partition on
    the real line is

        (-∞, X_t'β] ∪ (X_t'β, X_t'β + δ] ∪ ... ∪ (X_t'β + J_t δ, ∞)

    where J_t = D_t Y peers of t take action in Y (for y_t=1; for y_t=0
    the sub-bucket lives in the rightmost half-line). Given the drawn U,
    the specific scenario b is the unique combination of per-agent sub-
    buckets containing U, and the bucket boundaries enter the importance-
    weight ratio (not the broader sampling range).

    Under scenario recycling (Appendix D) the sub-bucket boundaries are
    affine in theta — each agent's bound is X_t'β + k_t · δ with k_t an
    integer fixed at sampling time. We cache those integer multipliers
    (`coef_lo`, `coef_hi`; -inf/+inf sentinels where a side is unbounded)
    plus the scenario-sampling log-weight log ω_sum at θ^{(0)}, which is
    the constant denominator of the importance weight.

    Returns:
        U         (T,)  drawn shocks
        coef_lo   (T,)  integer (or -inf) δ-multiplier for the sub-bucket lower
        coef_hi   (T,)  integer (or +inf) δ-multiplier for the sub-bucket upper
        log_mass0 float log Π_t [F(b̄_t(θ^{(0)})) - F(b̌_t(θ^{(0)}))]
                         (per-scenario log-mass at sampling theta)
        log_omega float log Π_t ω_t(θ^{(0)}), the sampling-truncation factor
                         (paper Eq 11); scenario-specific but constant in θ.
    """
    T = X.shape[0]
    Xb = X @ beta
    U = np.empty(T)
    coef_lo = np.full(T, -np.inf)
    coef_hi = np.full(T,  np.inf)
    log_omega = 0.0

    Y_f = Y.astype(float)
    DY = D.astype(float) @ Y_f              # D_t Y for each t (precomputed)
    J  = D.sum(axis=1).astype(int)          # total peers (degree)

    # ------------------------------------------------------------------
    # Step 2: y_t = 0 — sampling range (X_t'β + δ·D_t Y, ∞).
    # Paper's ω_t = 1/[1 − F(X_t'β + δ·D_t Y)].
    # Step 3.b identifies the specific sub-bucket m ∈ {D_t Y, ..., J_t}:
    #     U_t ∈ (X_t'β + m δ, X_t'β + (m+1) δ]     (m < J_t)
    #     U_t ∈ (X_t'β + J_t δ, ∞)                  (m = J_t)
    # ------------------------------------------------------------------
    for t in range(T):
        if not Y[t]:
            d_t = float(DY[t])
            lo = Xb[t] + delta * d_t
            U_t = _truncated_normal_above(lo, rng)
            U[t] = U_t
            log_omega -= norm.logsf(lo)            # −log[1 − F(lo)] = log ω_t

            # Step 3.b for y_t = 0.
            k_min = int(round(d_t))
            J_t = int(J[t])
            if delta <= 0 or U_t > Xb[t] + J_t * delta:
                # Top half-line sub-bucket (J_t, ∞) in δ-units.
                coef_lo[t] = float(J_t)
                coef_hi[t] = np.inf
            else:
                m = int(np.floor((U_t - Xb[t]) / delta))
                m = max(m, k_min)           # can't be below the enforced lower
                m = min(m, J_t - 1)         # finite bucket tops at J_t·δ
                coef_lo[t] = float(m)
                coef_hi[t] = float(m + 1)

    # ------------------------------------------------------------------
    # Step 3: y_t = 1 — sampling range (−∞, h_t] from Threshold Finder.
    # Paper's ω_t = 1/F(h_t). After drawing U_t, Step 3.b identifies the
    # sub-bucket: k_t ∈ {0, 1, ..., D_t Y} such that
    #     U_t ∈ (X_t'β + (k_t − 1) δ, X_t'β + k_t δ]     if k_t ≥ 1
    #     U_t ∈ (−∞, X_t'β]                               if k_t = 0
    # ------------------------------------------------------------------
    for t in range(T):
        if Y[t]:
            h, _Ytilde = _threshold_finder(X, D, Y, beta, delta, U, t)
            U_t = _truncated_normal_below(h, rng)
            U[t] = U_t
            log_omega -= norm.logcdf(h)            # −log F(h_t)

            # Step 3.b: which sub-bucket does U_t land in?
            # k_t = 0 iff U_t ≤ X_t'β; otherwise k_t = ceil((U_t − X_t'β) / δ)
            # clamped above by D_t Y (all other values inconsistent with
            # Y being the min NE under the Threshold Finder's guarantees).
            k_max = int(round(DY[t]))
            if delta > 0 and U_t > Xb[t]:
                k_t = int(np.ceil((U_t - Xb[t]) / delta))
                k_t = min(max(k_t, 1), k_max)
                coef_lo[t] = float(k_t - 1)
                coef_hi[t] = float(k_t)
            else:
                # Strict dominant sub-bucket (−∞, X_t'β]; coef is 0 for the
                # upper boundary (X_t'β + 0·δ), and lower is −∞.
                coef_lo[t] = -np.inf
                coef_hi[t] = 0.0

    # Log-mass at sampling θ of the SPECIFIC sub-bucket per agent.
    lowers0 = np.where(np.isfinite(coef_lo), Xb + delta * coef_lo, -np.inf)
    uppers0 = np.where(np.isfinite(coef_hi), Xb + delta * coef_hi,  np.inf)
    log_mass0 = float(_log_cdf_interval(lowers0, uppers0).sum())

    return U, coef_lo, coef_hi, log_mass0, log_omega


def sample_scenario(X, D, Y, beta, delta, rng, selection="min"):
    """Public entry point. Returns (U, coef_lo, coef_hi, log_mass0, log_omega).
    See ``_sample_scenario_min`` for field semantics."""
    if selection != "min":
        raise NotImplementedError(
            "Only 'min' selection is implemented; max-NE variant requires "
            "re-deriving Alg 1 (the simple sign-flip duality fails for "
            "graphs with degree heterogeneity).")
    return _sample_scenario_min(X, D, Y, beta, delta, rng)


# ---------------------------------------------------------------------------
# CRN scenario sampler (paper Appendix D first paragraph)
# ---------------------------------------------------------------------------

def _minimal_ne_fast(D_f, U, Xb, delta, T):
    """Hot-path minimal-NE iteration. D_f is float adjacency, Xb = X@beta."""
    y = np.zeros(T, dtype=bool)
    for _ in range(T + 1):
        margin = Xb + delta * (D_f @ y.astype(np.float64)) - U
        y_new = margin >= 0.0
        if np.array_equal(y_new, y):
            return y
        y = y_new
    return y


def _threshold_finder_fast(D_f, Y_bool, U, Xb, delta, t, idx_arange,
                            DY_at_t, T):
    """Vectorized threshold finder for the CRN inner loop."""
    use_U = (idx_arange < t) | (~Y_bool)
    Utilde = np.where(use_U, U, Xb - 1.0)
    Utilde[t] = Xb[t] + delta * DY_at_t + 1.0
    Ytilde = _minimal_ne_fast(D_f, Utilde, Xb, delta, T)
    DY_tilde_at_t = float(D_f[t] @ Ytilde.astype(np.float64))
    h_t = Xb[t] + delta * DY_tilde_at_t
    return h_t


def sample_scenario_at_theta(X, D, Y, beta, delta, uniforms_t):
    """Sample scenario at the current θ using fixed standard uniforms.

    Implements paper's Appendix D first-paragraph protocol: pre-drawn
    standard uniforms u_t ~ U(0,1) are held fixed; at each θ during
    optimization the scenario is re-derived via inverse-CDF on the
    truncated normal densities defined by Algorithm 3 at the current θ.

    The IS proposal moves with the optimizer (proposal = λ_y(·;θ) at the
    current θ, not at a fixed anchor θ⁽⁰⁾). The IS ratio collapses to
    ζ/λ = ∏ₜ 1/ωₜ(θ); we return log Πₜ ωₜ(θ).

    Args:
        uniforms_t: (T,) array of U(0,1) values, one per agent.

    Returns:
        log_omega : float, log Πₜ ωₜ(θ).
    """
    T = X.shape[0]
    Xb = X @ beta
    Y_bool = Y.astype(bool)
    Y_f = Y_bool.astype(np.float64)
    D_f = D.astype(np.float64) if D.dtype != np.float64 else D
    DY  = D_f @ Y_f                      # used for both y_t=0 lo and Step 2 of TF
    U   = np.empty(T)
    idx = np.arange(T)
    log_omega = 0.0

    # Step 2: y_t = 0 — vectorized lo + inverse-CDF; loop only over y_t=0
    # entries, but quantile arithmetic is element-wise.
    is_y0 = ~Y_bool
    if is_y0.any():
        lo_y0 = Xb[is_y0] + delta * DY[is_y0]
        F_lo = norm.cdf(lo_y0)
        u0 = uniforms_t[is_y0]
        q = F_lo + u0 * (1.0 - F_lo)
        U[is_y0] = norm.ppf(np.clip(q, 1e-15, 1 - 1e-15))
        log_omega -= norm.logsf(lo_y0).sum()

    # Step 3: y_t = 1 — threshold finder per agent, sequential (each call
    # depends on previously-drawn U entries for y_{t'<t} = 1 agents).
    y1_idx = np.flatnonzero(Y_bool)
    for t in y1_idx:
        h = _threshold_finder_fast(
            D_f, Y_bool, U, Xb, delta, int(t), idx, float(DY[t]), T)
        F_h = norm.cdf(h)
        q = uniforms_t[t] * F_h
        U[t] = norm.ppf(np.clip(q, 1e-15, 1 - 1e-15))
        log_omega -= norm.logcdf(h)

    return log_omega


# ---------------------------------------------------------------------------
# Truncated-normal samplers (numerically stable)
# ---------------------------------------------------------------------------

def _truncated_normal_above(lo, rng):
    """Draw U ~ N(0,1) conditional on U > lo."""
    # Inverse CDF: U = F^{-1}(F(lo) + V*(1 - F(lo))), V ~ U(0,1)
    F_lo = norm.cdf(lo)
    v = rng.uniform(0.0, 1.0)
    q = F_lo + v * (1.0 - F_lo)
    return norm.ppf(np.clip(q, 1e-15, 1 - 1e-15))


def _truncated_normal_below(hi, rng):
    """Draw U ~ N(0,1) conditional on U <= hi."""
    F_hi = norm.cdf(hi)
    v = rng.uniform(0.0, 1.0)
    q = v * F_hi
    return norm.ppf(np.clip(q, 1e-15, 1 - 1e-15))


def _log_cdf_interval(lowers, uppers):
    """Numerically stable log(F(up) - F(lo)) elementwise."""
    # scipy provides log_ndtr via scipy.special, but norm.logcdf is fine here.
    # For narrow intervals we fall back to log(exp(logcdf(up)) - exp(logcdf(lo))).
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
            # F(up) - F(lo) with care
            a, b = norm.cdf(lo), norm.cdf(up)
            out[i] = np.log(max(b - a, 1e-300))
    return out
