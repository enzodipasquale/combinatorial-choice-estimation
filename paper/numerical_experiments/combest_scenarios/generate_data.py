#!/usr/bin/env python3
import numpy as np


def generate_item_characteristics(M, rho=0.5, seed=None):
    rng = np.random.default_rng(seed)
    xi = rng.normal(0, 1, M)
    z = rng.normal(0, 1, (M, 3))
    phi = z + rho * xi[:, None]
    return phi, z, xi


def generate_delta_star(phi, beta_star, xi, target_std=1.0):
    delta = phi @ beta_star + xi
    delta = delta - delta.mean()                    # demean
    if delta.std() > 0:
        delta = delta / delta.std() * target_std    # normalize spread
    return delta


def generate_gross_substitutes_data(N, M, alpha, lambda_gs, delta_star, phi, z, seed=None):
    rng = np.random.default_rng(seed)
    modular_agent = rng.normal(0, 1, (N, M, 1))

    quadratic_feature = np.zeros((M, M, 1))
    for j in range(M):
        for jp in range(M):
            if j != jp:
                quadratic_feature[j, jp, 0] = -1.0

    delta_adjusted = delta_star - lambda_gs

    return {
        "id_data": {
            "modular": modular_agent,
            "obs_bundles": np.zeros((N, M), dtype=bool),
        },
        "item_data": {
            "modular": -np.eye(M),
            "quadratic": quadratic_feature,
        },
        "meta": {
            "phi": phi, "z": z,
            "delta_star": delta_adjusted,
            "alpha_star": np.array([alpha]),
            "lambda_star": np.array([lambda_gs]),
        },
    }


def generate_supermodular_data(N, M, alpha, lambda_quad, delta_star, phi, z, seed=None):
    rng = np.random.default_rng(seed)
    modular_agent = rng.normal(0, 1, (N, M, 1))

    quadratic_agent = np.zeros((N, M, M, 1))
    for i in range(N):
        for j in range(M):
            for jp in range(j + 1, M):
                x_val = rng.binomial(1, 0.5)
                quadratic_agent[i, j, jp, 0] = x_val
                quadratic_agent[i, jp, j, 0] = x_val

    return {
        "id_data": {
            "modular": modular_agent,
            "quadratic": quadratic_agent,
            "obs_bundles": np.zeros((N, M), dtype=bool),
        },
        "item_data": {
            "modular": -np.eye(M),
            "quadratic": np.zeros((M, M, 0)),
        },
        "meta": {
            "phi": phi, "z": z,
            "delta_star": delta_star,
            "alpha_star": np.array([alpha]),
            "lambda_star": np.array([lambda_quad]),
        },
    }


def generate_linear_knapsack_data(N, M, alpha, delta_star, phi, z, seed=None):
    rng = np.random.default_rng(seed)
    modular_agent = rng.normal(0, 1, (N, M, 1))
    weights = rng.uniform(0.5, 1.5, M)
    capacities = rng.uniform(0.3 * weights.sum(), 0.5 * weights.sum(), N)

    return {
        "id_data": {
            "modular": modular_agent,
            "capacity": capacities,
            "obs_bundles": np.zeros((N, M), dtype=bool),
        },
        "item_data": {
            "modular": -np.eye(M),
            "weight": weights,
        },
        "meta": {
            "phi": phi, "z": z,
            "delta_star": delta_star,
            "alpha_star": np.array([alpha]),
            "lambda_star": np.array([]),
        },
    }


def generate_quadratic_knapsack_data(N, M, alpha, lambda_quad, delta_star, phi, z, seed=None):
    rng = np.random.default_rng(seed)

    quadratic_item = np.zeros((M, M, 1))
    for j in range(M):
        for jp in range(j + 1, M):
            x_val = rng.binomial(1, 0.5)
            quadratic_item[j, jp, 0] = x_val
            quadratic_item[jp, j, 0] = x_val

    modular_agent = rng.normal(0, 1, (N, M, 1))
    weights = rng.uniform(0.5, 1.5, M)
    capacities = rng.uniform(0.3 * weights.sum(), 0.5 * weights.sum(), N)

    return {
        "id_data": {
            "modular": modular_agent,
            "capacity": capacities,
            "obs_bundles": np.zeros((N, M), dtype=bool),
        },
        "item_data": {
            "modular": -np.eye(M),
            "quadratic": quadratic_item,
            "weight": weights,
        },
        "meta": {
            "phi": phi, "z": z,
            "delta_star": delta_star,
            "alpha_star": np.array([alpha]),
            "lambda_star": np.array([lambda_quad]),
        },
    }


def generate_data(spec, N, M, alpha=None, lambda_val=None, rho=0.5, beta_star=None, seed=None):
    rng = np.random.default_rng(seed)
    phi, z, xi = generate_item_characteristics(M, rho, seed=rng.integers(0, 2**31))

    if beta_star is None:
        beta_star = np.ones(phi.shape[1])
    delta_star = generate_delta_star(phi, beta_star, xi)
    alpha = alpha if alpha is not None else 0.1

    subseed = rng.integers(0, 2**31)

    if spec == "gross_substitutes":
        lam = lambda_val or 0.1
        input_data = generate_gross_substitutes_data(N, M, alpha, lam, delta_star, phi, z, seed=subseed)
        theta_star = np.concatenate([[alpha], input_data["meta"]["delta_star"], [lam]])
    elif spec == "supermodular":
        lam = lambda_val or 0.05
        input_data = generate_supermodular_data(N, M, alpha, lam, delta_star, phi, z, seed=subseed)
        theta_star = np.concatenate([[alpha], delta_star, [lam]])
    elif spec == "linear_knapsack":
        input_data = generate_linear_knapsack_data(N, M, alpha, delta_star, phi, z, seed=subseed)
        theta_star = np.concatenate([[alpha], delta_star])
    elif spec == "quadratic_knapsack":
        lam = lambda_val or 0.05
        input_data = generate_quadratic_knapsack_data(N, M, alpha, lam, delta_star, phi, z, seed=subseed)
        theta_star = np.concatenate([[alpha], delta_star, [lam]])
    else:
        raise ValueError(f"Unknown specification: {spec}")

    has_lambda = spec != "linear_knapsack"
    input_data["meta"]["alpha_indices"] = [0]
    input_data["meta"]["delta_indices"] = list(range(1, M + 1))
    input_data["meta"]["lambda_indices"] = [M + 1] if has_lambda else []

    return input_data, theta_star
