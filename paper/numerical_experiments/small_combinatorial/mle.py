#!/usr/bin/env python3
"""GHK-based simulated MLE for combinatorial choice at small J.

Uses analytical gradients (chain rule through Φ) for L-BFGS-B optimization.
"""
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


def ghk_ll_and_grad(theta, features, all_bundles, obs_bundle_idx,
                    feasibility_mask, sigma, uniform_draws):
    """
    GHK simulated log-likelihood AND analytical gradient w.r.t. theta.

    For agent i with observed bundle S_i, GHK computes P(S_i chosen | theta) as:
      P̂_i = (1/R) sum_r prod_j p_j^(r)(theta)

    where p_j^(r) = Phi(u_j^(r)) - Phi(l_j^(r)) depends on theta through:
      - b(theta) = (features[i, alt] - features[i, obs]) @ theta  (constraint RHS)
      - residual updates from previous truncated draws

    Gradient: d(log P̂_i)/dtheta = E_w[d(log_prob_r)/dtheta]
    where w_r proportional to exp(log_prob_r) (IS weights).

    For each simulation r, d(log_prob_r)/dtheta is computed by propagating
    d(residual)/dtheta through the sequential steps using the chain rule.

    Returns:
        ll: scalar log-likelihood
        grad: (K,) gradient
    """
    N = features.shape[0]
    J = all_bundles.shape[1]
    n_bundles = len(all_bundles)
    R = uniform_draws.shape[0]
    K = len(theta)
    bundles_float = all_bundles.astype(float)

    det_util = features @ theta  # (N, 2^J)

    ll = 0.0
    grad = np.zeros(K)

    for i in range(N):
        obs_idx = obs_bundle_idx[i]
        obs_bundle = bundles_float[obs_idx]  # (J,)

        if feasibility_mask is not None:
            valid = feasibility_mask[i].copy()
        else:
            valid = np.ones(n_bundles, dtype=bool)
        valid[obs_idx] = False

        alt_indices = np.where(valid)[0]
        n_alt = len(alt_indices)
        if n_alt == 0:
            continue

        # A[k, j] = S_i[j] - S'_k[j]:  constraint coefficients
        A = obs_bundle[None, :] - bundles_float[alt_indices]  # (n_alt, J)
        # b[k] = det_util[i, alt_k] - det_util[i, obs]:  constraint RHS
        b = det_util[i, alt_indices] - det_util[i, obs_idx]  # (n_alt,)
        # D[k, :] = features[i, alt_k, :] - features[i, obs, :]:  d(b)/d(theta)
        D = features[i, alt_indices, :] - features[i, obs_idx, :]  # (n_alt, K)

        # --- Initialize simulation state ---
        residual = np.tile(b, (R, 1))          # (R, n_alt)
        # d(residual[r,k])/d(theta): same init D for all r
        d_residual = np.tile(D, (R, 1, 1))     # (R, n_alt, K)

        log_probs = np.zeros(R)
        d_log_probs = np.zeros((R, K))

        for j in range(J):
            a_j = A[:, j]           # (n_alt,)
            pos_mask = a_j > 0.5    # a_j = +1
            neg_mask = a_j < -0.5  # a_j = -1

            # --- Lower bound: max over pos_mask constraints ---
            if pos_mask.any():
                lb_vals = residual[:, pos_mask] / sigma  # (R, n_pos)
                lower = lb_vals.max(axis=1)              # (R,)
                argmax_l = lb_vals.argmax(axis=1)        # (R,) index into pos_mask
                k_star = np.where(pos_mask)[0][argmax_l] # (R,) index into n_alt
                d_lower = d_residual[np.arange(R), k_star, :] / sigma  # (R, K)
            else:
                lower = np.full(R, -np.inf)
                d_lower = np.zeros((R, K))

            # --- Upper bound: min over neg_mask constraints ---
            if neg_mask.any():
                ub_vals = -residual[:, neg_mask] / sigma  # (R, n_neg)
                upper = ub_vals.min(axis=1)               # (R,)
                argmin_u = ub_vals.argmin(axis=1)         # (R,) index into neg_mask
                k_dstar = np.where(neg_mask)[0][argmin_u] # (R,) index into n_alt
                d_upper = -d_residual[np.arange(R), k_dstar, :] / sigma  # (R, K)
            else:
                upper = np.full(R, np.inf)
                d_upper = np.zeros((R, K))

            # --- Probability contribution ---
            cdf_l = norm.cdf(lower)   # Phi(lower)
            cdf_u = norm.cdf(upper)   # Phi(upper)
            p_j = np.maximum(cdf_u - cdf_l, 1e-15)
            log_probs += np.log(p_j)

            phi_l = norm.pdf(lower)   # phi(lower), 0 when lower=-inf
            phi_u = norm.pdf(upper)   # phi(upper), 0 when upper=+inf

            # d(log p_j)/d(theta) = [phi_u * d_upper - phi_l * d_lower] / p_j
            d_log_probs += (phi_u[:, None] * d_upper - phi_l[:, None] * d_lower) / p_j[:, None]

            # --- Draw eps_j from truncated N(0, sigma^2) on [sigma*lower, sigma*upper] ---
            u = uniform_draws[:, j]                             # (R,)
            cdf_val = cdf_l + u * p_j
            cdf_val = np.clip(cdf_val, 1e-10, 1 - 1e-10)
            eta_j = norm.ppf(cdf_val)                           # standardized draw (R,)
            eps_j = sigma * eta_j                               # (R,)

            # --- Gradient of eps_j w.r.t. theta ---
            # d(cdf_val)/d(theta) = (1-u)*phi_l*d_lower + u*phi_u*d_upper
            # d(eps_j)/d(theta) = sigma/phi(eta_j) * d(cdf_val)/d(theta)
            phi_eta = np.maximum(norm.pdf(eta_j), 1e-15)       # (R,)
            d_cdf = ((1 - u[:, None]) * phi_l[:, None] * d_lower
                     + u[:, None] * phi_u[:, None] * d_upper)  # (R, K)
            d_eps_j = sigma * d_cdf / phi_eta[:, None]          # (R, K)

            # --- Propagate residual gradient ---
            # d(residual[r,k])/d(theta) -= a_j[k] * d(eps_j)/d(theta)
            d_residual -= a_j[None, :, None] * d_eps_j[:, None, :]

            # --- Update residual ---
            residual -= a_j[None, :] * eps_j[:, None]

        # --- Aggregate over R draws ---
        # log P̂_i = log(mean_r exp(log_probs))
        max_lp = log_probs.max()
        w = np.exp(log_probs - max_lp)              # (R,) unnormalized IS weights
        sum_w = w.sum()
        prob = sum_w / R
        ll += np.log(max(prob, 1e-15)) + max_lp

        # d(log P̂_i)/d(theta) = weighted mean of d_log_probs_r
        grad += (w @ d_log_probs) / max(sum_w, 1e-30)

    return ll, grad


def estimate_smle(input_data, obs_bundles, sigma=1.0, R=500, seed=42,
                  theta0=None, maxiter=2000, bounds=None):
    """
    Estimate parameters by GHK-based simulated MLE with analytical gradients.

    Uses L-BFGS-B with exact gradient (chain rule through Phi functions).

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

    # Pre-generate uniform draws for GHK inverse CDF sampling (fixed across evaluations)
    rng = np.random.default_rng(seed)
    uniform_draws = rng.uniform(0, 1, (R, J))

    if theta0 is None:
        theta0 = np.zeros(K)

    def neg_ll_and_grad(theta):
        ll, g = ghk_ll_and_grad(theta, features, all_bundles, obs_bundle_idx,
                                 feasibility_mask, sigma, uniform_draws)
        return -ll, -g

    result = minimize(
        neg_ll_and_grad, theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )

    return result.x, result
