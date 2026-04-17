#!/usr/bin/env python3
"""Simulated MLE for combinatorial choice at small J.

Uses a smoothed IS estimator: for each draw of item-level errors, compute
log P_smooth = log_softmax(bundle_utils / tau)[S*]. The simulated log-likelihood is
logsumexp over draws (not mean of log), giving an unbiased estimator of log P.

Why not GHK: GHK sequential conditioning on eps_j is biased for bundles with
multiple items because it over-constrains eps_j by ignoring future eps's
compensating contribution (e.g., for the empty bundle constraint with a=(1,1),
GHK imposes eps_1 <= -3 when only eps_1+eps_2 <= -3 is required). This causes
catastrophic probability underestimates for rare observed bundles (>38x bias at J=4).

This implementation uses:
  log P̂_i(θ) = logsumexp_r(log_softmax(util^r/τ)[S_i*]) - log R
  ∂ log P̂_i / ∂θ = Σ_r w_r * (1/τ) * (features[i,S*] - E_{P^r}[features[i,S]])
where w_r = IS weights = softmax(log_P_smooth^r).
As τ→0, log P̂_i → log P̂_freq → log P_true (frequency simulator limit).
"""
import numpy as np
from scipy.optimize import minimize


def enumerate_bundles(J):
    """All 2^J bundles as boolean arrays."""
    return np.array([[bool((i >> j) & 1) for j in range(J)] for i in range(2**J)])


def build_features(input_data, all_bundles):
    """Compute covariate features (N, 2^J, K) matching combest's oracle."""
    id_data = input_data["id_data"]
    item_data = input_data["item_data"]
    N = next(v for v in id_data.values() if hasattr(v, 'ndim') and v.ndim >= 2).shape[0]
    n_bundles = len(all_bundles)
    bundles_float = all_bundles.astype(float)

    feat_list = []
    if "modular" in id_data:
        feat_list.append(np.einsum("ijk,bj->ibk", id_data["modular"], bundles_float))
    if "modular" in item_data:
        f = np.einsum("jk,bj->bk", item_data["modular"], bundles_float)
        feat_list.append(np.broadcast_to(f[None], (N, n_bundles, f.shape[1])).copy())
    if "quadratic" in id_data:
        feat_list.append(np.einsum("ijlk,bj,bl->ibk", id_data["quadratic"], bundles_float, bundles_float))
    if "quadratic" in item_data and item_data["quadratic"].shape[-1] > 0:
        f = np.einsum("jlk,bj,bl->bk", item_data["quadratic"], bundles_float, bundles_float)
        feat_list.append(np.broadcast_to(f[None], (N, n_bundles, f.shape[1])).copy())

    return np.concatenate(feat_list, axis=-1)


def build_feasibility_mask(input_data, all_bundles):
    """Feasibility mask (N, 2^J) for knapsack constraints."""
    id_data = input_data["id_data"]
    item_data = input_data["item_data"]
    if "weight" not in item_data or "capacity" not in id_data:
        return None
    weights = item_data["weight"]
    capacities = id_data["capacity"]
    bundle_weights = all_bundles.astype(float) @ weights
    return bundle_weights[None, :] <= capacities[:, None]


def smooth_is_ll_and_grad(theta, features, all_bundles, obs_bundle_idx,
                           feasibility_mask, sigma, eps_draws, tau):
    """
    Smoothed IS log-likelihood and analytical gradient.

    Per-draw log probability (smooth):
      log_p_r = log_softmax(util^r / tau)[S*]
      util^r[S] = features[i,S] @ theta + sum_{j in S} eps_r[j]

    Simulated log-likelihood (logsumexp, not mean-of-log):
      log P̂_i = logsumexp_r(log_p_r) - log R

    Gradient via IS weights w_r = softmax(log_p_r):
      ∂ log P̂_i / ∂θ = (1/tau) * Σ_r w_r * [features[i,S*] - E_{P^r}[features]]

    As tau -> 0: log P̂_i -> log(fraction of draws where S* = argmax) = log P_freq -> log P_true.
    The bias from finite tau is O(tau) and negligible for tau=0.1.

    Parameters
    ----------
    eps_draws : (R, J) pre-drawn N(0, sigma^2) errors, fixed during optimization
    tau : temperature, default 0.1
    """
    N = features.shape[0]
    R = eps_draws.shape[0]
    K = len(theta)
    n_bundles = len(all_bundles)
    bundles_float = all_bundles.astype(float)

    det_util = features @ theta          # (N, 2^J)
    stoch = bundles_float @ eps_draws.T  # (2^J, R): stoch[S,r] = sum_{j in S} eps_r[j]

    ll = 0.0
    grad = np.zeros(K)

    for i in range(N):
        obs_idx = obs_bundle_idx[i]

        if feasibility_mask is not None:
            valid = feasibility_mask[i]
        else:
            valid = np.ones(n_bundles, dtype=bool)
        valid_idx = np.where(valid)[0]

        obs_pos = np.searchsorted(valid_idx, obs_idx)  # position of S* in valid

        # Bundle utilities: (n_valid, R)
        util = det_util[i, valid_idx, None] + stoch[valid_idx, :]

        # Per-draw log softmax at S*
        util_tau = util / tau                           # (n_valid, R)
        max_u = util_tau.max(axis=0, keepdims=True)    # (1, R)
        log_sum_exp = np.log(np.exp(util_tau - max_u).sum(axis=0)) + max_u[0]  # (R,)
        log_p_r = util_tau[obs_pos, :] - log_sum_exp   # (R,) per-draw log probs

        # log P̂_i = logsumexp(log_p_r) - log R
        max_lp = log_p_r.max()
        sum_exp = np.exp(log_p_r - max_lp).sum()
        ll += np.log(sum_exp) + max_lp - np.log(R)

        # IS weights: w_r = softmax(log_p_r)
        w = np.exp(log_p_r - max_lp)
        w /= sum_exp   # (R,) normalized IS weights

        # Gradient: Σ_r w_r * (1/tau) * [features[S*] - E_{P^r}[features]]
        # E_{P^r}[features] = Σ_S P_smooth^r(S) * features[i,S]
        P_smooth = np.exp(util_tau - log_sum_exp[None, :])  # (n_valid, R)
        feat_valid = features[i, valid_idx, :]               # (n_valid, K)
        # IS-weighted expected features: (K,)
        # P_smooth @ w: (n_valid,) — per-bundle IS-weighted prob sum
        exp_feat_IS = feat_valid.T @ (P_smooth @ w)  # (K,)
        # feat[S*] IS-weighted (constant across r, so just features[i,obs_idx])
        grad += (features[i, obs_idx, :] - exp_feat_IS) / tau

    return ll, grad


def estimate_smle(input_data, obs_bundles, sigma=1.0, R=500, seed=42,
                  theta0=None, maxiter=2000, bounds=None, tau=0.1):
    """
    Estimate parameters by smoothed IS simulated MLE with analytical gradients.

    Parameters
    ----------
    tau : smoothing temperature. 0.1 gives negligible bias for typical utility scales.
          Use 0.05 if utility differences are small (sigma comparable to det_util spread).
    """
    J = obs_bundles.shape[1]
    all_bundles = enumerate_bundles(J)

    obs_bundle_idx = np.zeros(len(obs_bundles), dtype=int)
    for i, ob in enumerate(obs_bundles):
        for b_idx, b in enumerate(all_bundles):
            if np.array_equal(ob, b):
                obs_bundle_idx[i] = b_idx
                break

    features = build_features(input_data, all_bundles)
    K = features.shape[2]
    feasibility_mask = build_feasibility_mask(input_data, all_bundles)

    rng = np.random.default_rng(seed)
    eps_draws = rng.normal(0, sigma, (R, J))

    if theta0 is None:
        theta0 = np.zeros(K)

    def neg_ll_and_grad(theta):
        ll, g = smooth_is_ll_and_grad(theta, features, all_bundles, obs_bundle_idx,
                                       feasibility_mask, sigma, eps_draws, tau)
        return -ll, -g

    result = minimize(
        neg_ll_and_grad, theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )

    return result.x, result
