#!/usr/bin/env python3
"""GHK-based simulated MLE for combinatorial choice at small J."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def enumerate_bundles(J):
    """All 2^J bundles as boolean arrays, including empty bundle."""
    return np.array([[bool((i >> j) & 1) for j in range(J)] for i in range(2**J)])


def build_features(input_data, all_bundles):
    """Precompute covariate features for all (agent, bundle) pairs.
    Returns: features (N, 2^J, K) array."""
    id_data = input_data["id_data"]
    item_data = input_data["item_data"]
    N = id_data["modular"].shape[0]
    n_bundles = len(all_bundles)
    bundles_float = all_bundles.astype(float)

    feat_list = []
    if "modular" in id_data:
        f = np.einsum("ijk,bj->ibk", id_data["modular"], bundles_float)
        feat_list.append(f)
    if "modular" in item_data:
        f_item = np.einsum("jk,bj->bk", item_data["modular"], bundles_float)
        feat_list.append(np.broadcast_to(f_item[None, :, :], (N, n_bundles, f_item.shape[1])).copy())
    if "quadratic" in id_data:
        f = np.einsum("ijlk,bj,bl->ibk", id_data["quadratic"], bundles_float, bundles_float)
        feat_list.append(f)
    if "quadratic" in item_data and item_data["quadratic"].shape[-1] > 0:
        f_qi = np.einsum("jlk,bj,bl->bk", item_data["quadratic"], bundles_float, bundles_float)
        feat_list.append(np.broadcast_to(f_qi[None, :, :], (N, n_bundles, f_qi.shape[1])).copy())

    return np.concatenate(feat_list, axis=-1)


def build_feasibility_mask(input_data, all_bundles):
    """For knapsack specs, mark infeasible bundles per agent."""
    id_data = input_data["id_data"]
    item_data = input_data["item_data"]
    if "weight" not in item_data or "capacity" not in id_data:
        return None
    weights = item_data["weight"]
    capacities = id_data["capacity"]
    bundle_weights = all_bundles.astype(float) @ weights
    return bundle_weights[None, :] <= capacities[:, None]


def ghk_log_likelihood(theta, features, all_bundles, obs_bundle_idx,
                       feasibility_mask, sigma, uniform_draws):
    """
    Vectorized GHK-based simulated log-likelihood.

    For each agent i with observed bundle S_i, compute P(S_i chosen | theta)
    using the GHK simulator on the polytope {eps : v(S_i, eps) >= v(S', eps) for all S'}.

    The error for bundle S is sum_{j in S} eps_j where eps ~ N(0, sigma^2 I_J).
    Constraint v(S) >= v(S'): a'eps >= b where a = 1_{j in S} - 1_{j in S'}.

    GHK conditions sequentially on eps_1, ..., eps_J:
    - At step j, find tightest lower bound on eps_j from constraints
    - P_j = 1 - Phi(lower_j / sigma)
    - Draw eps_j from truncated normal
    - P(S chosen) = product of P_j, averaged over R draws

    Args:
        theta: (K,) parameters
        features: (N, 2^J, K) precomputed
        all_bundles: (2^J, J) boolean
        obs_bundle_idx: (N,) int
        feasibility_mask: (N, 2^J) or None
        sigma: scalar
        uniform_draws: (R, J) uniform(0,1) draws for inverse CDF sampling
    """
    N = features.shape[0]
    J = all_bundles.shape[1]
    n_bundles = len(all_bundles)
    R = uniform_draws.shape[0]
    bundles_float = all_bundles.astype(float)  # (2^J, J)

    det_util = features @ theta  # (N, 2^J)

    # For each agent, compute constraint coefficients and RHS
    # A_i[k, j] = S_i[j] - S'_k[j] for each alternative bundle S'_k
    # b_i[k] = det_util[i, k] - det_util[i, obs_idx]  (RHS, depends on theta)

    ll = 0.0

    for i in range(N):
        obs_idx = obs_bundle_idx[i]
        obs_bundle = bundles_float[obs_idx]  # (J,)

        # Determine valid alternatives
        if feasibility_mask is not None:
            valid = feasibility_mask[i].copy()
        else:
            valid = np.ones(n_bundles, dtype=bool)
        valid[obs_idx] = False  # exclude observed bundle

        alt_indices = np.where(valid)[0]
        if len(alt_indices) == 0:
            continue

        # A: (n_alt, J), b: (n_alt,)
        A = obs_bundle[None, :] - bundles_float[alt_indices]  # (n_alt, J)
        b = det_util[i, alt_indices] - det_util[i, obs_idx]   # (n_alt,)

        # Vectorized GHK over R draws
        # residual: (R, n_alt) - remaining RHS after conditioning
        residual = np.tile(b, (R, 1))  # (R, n_alt)
        log_probs = np.zeros(R)

        for j in range(J):
            a_j = A[:, j]  # (n_alt,)

            # For constraints with a_j > 0: eps_j >= residual[r,k] / (sigma * a_j)
            # For constraints with a_j < 0: eps_j <= residual[r,k] / (sigma * a_j)  (upper bound)
            # For a_j == 0: constraint doesn't involve eps_j

            pos_mask = a_j > 0.5   # a_j = 1
            neg_mask = a_j < -0.5  # a_j = -1

            # Lower bounds from positive coefficients: (R, n_pos)
            if pos_mask.any():
                # lower = max over k of residual[r,k] / sigma  (since a_j = 1)
                lower_bounds = residual[:, pos_mask] / sigma  # (R, n_pos)
                lower = lower_bounds.max(axis=1)  # (R,)
            else:
                lower = np.full(R, -np.inf)

            # Upper bounds from negative coefficients: (R, n_neg)
            if neg_mask.any():
                # a_j = -1: -eps_j >= residual/sigma => eps_j <= -residual/sigma
                upper_bounds = -residual[:, neg_mask] / sigma  # (R, n_neg)
                upper = upper_bounds.min(axis=1)  # (R,)
            else:
                upper = np.full(R, np.inf)

            # P_j = Phi(upper) - Phi(lower)
            cdf_lower = norm.cdf(lower)
            cdf_upper = norm.cdf(upper)
            p_j = np.maximum(cdf_upper - cdf_lower, 1e-15)
            log_probs += np.log(p_j)

            # Draw eps_j from truncated N(0, sigma^2) on [sigma*lower, sigma*upper]
            # Use inverse CDF: eps_j = sigma * Phi^{-1}(cdf_lower + u * (cdf_upper - cdf_lower))
            u = uniform_draws[:, j]
            cdf_val = cdf_lower + u * (cdf_upper - cdf_lower)
            cdf_val = np.clip(cdf_val, 1e-10, 1 - 1e-10)
            eta_j = norm.ppf(cdf_val)  # standardized draw
            eps_j = sigma * eta_j

            # Update residual: residual -= a_j * eps_j
            residual -= a_j[None, :] * eps_j[:, None]

        # P(S_i chosen) = mean over R of exp(log_probs)
        # Use log-sum-exp for stability
        max_lp = log_probs.max()
        prob = np.exp(max_lp) * np.mean(np.exp(log_probs - max_lp))
        ll += np.log(max(prob, 1e-15))

    return ll


def estimate_smle(input_data, obs_bundles, sigma=1.0, R=500, seed=42,
                  theta0=None, maxiter=2000):
    """
    Estimate parameters by GHK-based simulated MLE.

    Returns:
        theta_hat: (K,) estimated parameters
        result: scipy OptimizeResult
    """
    J = obs_bundles.shape[1]
    all_bundles = enumerate_bundles(J)

    # Map observed bundles to indices
    obs_bundle_idx = np.zeros(len(obs_bundles), dtype=int)
    for i, ob in enumerate(obs_bundles):
        for b_idx, b in enumerate(all_bundles):
            if np.array_equal(ob, b):
                obs_bundle_idx[i] = b_idx
                break

    features = build_features(input_data, all_bundles)
    K = features.shape[2]
    feasibility_mask = build_feasibility_mask(input_data, all_bundles)

    # Pre-generate uniform draws for GHK inverse CDF sampling
    rng = np.random.default_rng(seed)
    uniform_draws = rng.uniform(0, 1, (R, J))

    if theta0 is None:
        theta0 = np.zeros(K)

    def neg_ll(theta):
        return -ghk_log_likelihood(theta, features, all_bundles, obs_bundle_idx,
                                    feasibility_mask, sigma, uniform_draws)

    # Use Nelder-Mead: GHK LL is fast to evaluate but numerical gradients
    # through simulation-based LL are noisy. Nelder-Mead is more robust.
    result = minimize(neg_ll, theta0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-5,
                               "adaptive": True})

    return result.x, result
