"""Scenario Sampler for the peer-effects game (Graham & Gonzalez 2023).

Implements the MINIMAL-NE variant (paper Algorithms 1 and 2 for the peer-
effects case). A MAX-NE variant would require re-deriving the sampler from
scratch — the simple "negate β and U" duality only works for degree-regular
graphs. Given this asymmetry, we run paper_sml on the min-NE DGP (paper's
native setting) and compare combest's max-NE numbers against the paper's
published Table 1 Panel B results.

Distribution: U_t ~ N(0, 1) iid (Appendix B). We use scipy.stats.norm.

Output of sample_scenario():
  U      : (T,) array of realized taste shocks
  bounds : (T, 2) array with per-player bucket [b_lower, b_upper] such that
           Pr(U_t in (b_lower, b_upper]) = F(b_upper) - F(b_lower). This is
           the scenario "bucket" needed to compute ζ(b; θ) exactly.
  weight : float, the importance weight 1 / λ_y(b; θ^{(0)}) evaluated along
           the sampling path (accumulated product of truncation factors).
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
    """Sample U such that Y is the minimal NE of the game at parameters θ.

    Returns (U, bounds, log_weight) where:
      bounds[t] = (lower, upper) with Pr(U_t ∈ (lower, upper]) = F(up) − F(lo)
      log_weight = sum_t log(F(upper_t) - F(lower_t))
                 = log λ_y(b; θ)   (in its truncated-distribution form)
    """
    T = X.shape[0]
    Xb = X @ beta
    U = np.empty(T)
    lowers = np.full(T, -np.inf)
    uppers = np.full(T, np.inf)

    # Step 2: agents with y_t = 0 — draw from truncated upper tail
    #   U_t ∈ (X_t'β + δ G_t Y , ∞)
    #   density f(u) / (1 − F(X_t'β + δ G_t Y))
    for t in range(T):
        if not Y[t]:
            lo = Xb[t] + delta * (D[t].astype(float) @ Y.astype(float))
            U[t] = _truncated_normal_above(lo, rng)
            lowers[t] = lo
            uppers[t] = np.inf

    # Step 3: agents with y_t = 1 — use Threshold Finder, then draw from
    #   (−∞, h_t] with density f(u) / F(h_t)
    for t in range(T):
        if Y[t]:
            h = _threshold_finder(X, D, Y, beta, delta, U, t)
            U[t] = _truncated_normal_below(h, rng)
            lowers[t] = -np.inf
            uppers[t] = h

    # Bucket log-mass log[F(upper) - F(lower)]
    log_mass = _log_cdf_interval(lowers, uppers)
    return U, lowers, uppers, float(log_mass.sum())


def sample_scenario(X, D, Y, beta, delta, rng, selection="min"):
    """Public entry point. Returns (U, lower, upper, log_mass)."""
    if selection != "min":
        raise NotImplementedError(
            "Only 'min' selection is implemented; max-NE variant requires "
            "re-deriving Alg 1 (the simple sign-flip duality fails for "
            "graphs with degree heterogeneity).")
    return _sample_scenario_min(X, D, Y, beta, delta, rng)


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
