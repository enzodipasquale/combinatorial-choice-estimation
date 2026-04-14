"""Oracle functions for the quadratic knapsack / auction scenario.

- compute_utility:      V_i(b) for one agent
- brute_force_demand:   exact optimizer (enumerate 2^M bundles), M<=15 only
- solve_qkp_gurobi:    exact QKP via Gurobi (MIPGap=0)
- twosls_smoke_test:    verify 2SLS recovers beta_star from delta_star
"""

import numpy as np
from itertools import product as cartesian_product


def compute_utility(bundle, x_i, delta, Q_dense, lambda_, alpha, errors_i):
    """Full utility V_i(b) for one agent (ignoring capacity — for evaluation).

    Quadratic term: lambda * sum_{j<j'} Q_{jj'} b_j b_{j'} = 0.5 * lambda * b^T Q b
    (factor 0.5 because Q is symmetric with zero diagonal).
    """
    b = bundle.astype(float)
    modular = alpha * x_i[bundle].sum() - delta[bundle].sum() + errors_i[bundle].sum()
    quadratic = 0.5 * lambda_ * b @ Q_dense @ b
    return modular + quadratic


def brute_force_demand(x_i, delta, Q_dense, lambda_, alpha, errors_i,
                       weights, capacity_i, M):
    """Enumerate all 2^M feasible bundles, return maximizer. M<=15 only."""
    best_val = float('-inf')
    best_bundle = np.zeros(M, dtype=bool)

    for bits in cartesian_product([False, True], repeat=M):
        b = np.array(bits, dtype=bool)
        # Capacity check
        if (weights[b].sum() if b.any() else 0) > capacity_i:
            continue
        val = compute_utility(b, x_i, delta, Q_dense, lambda_, alpha, errors_i)
        if val > best_val:
            best_val = val
            best_bundle = b.copy()

    return best_bundle, best_val


def solve_qkp_gurobi(x_i, delta, Q_dense, lambda_, alpha, errors_i,
                      weights, capacity_i, M, solver_cfg=None):
    """Solve single-agent QKP exactly via Gurobi.

    Returns (bundle, utility, gap, runtime).
    """
    import gurobipy as gp

    if solver_cfg is None:
        solver_cfg = {}

    model = gp.Model()
    model.setParam('OutputFlag', solver_cfg.get('OutputFlag', 0))
    model.setParam('MIPGap', solver_cfg.get('MIPGap', 0))
    if 'TimeLimit' in solver_cfg:
        model.setParam('TimeLimit', solver_cfg['TimeLimit'])

    b = model.addMVar(M, vtype=gp.GRB.BINARY, name='b')
    model.addConstr(weights @ b <= capacity_i)

    # Quadratic term: lambda * sum_{j<j'} Q_{jj'} b_j b_{j'}.
    # Gurobi computes b^T Q b = 2 * sum_{j<j'}, so pass 0.5 * lambda * Q.
    linear = alpha * x_i - delta + errors_i
    model.setMObjective(
        Q=0.5 * lambda_ * Q_dense,
        c=linear,
        constant=0.0,
        sense=gp.GRB.MAXIMIZE
    )
    model.optimize()

    gap = model.MIPGap if model.SolCount > 0 else float('inf')
    runtime = model.Runtime

    if model.SolCount > 0:
        bundle = np.array(model.x, dtype=bool)
        util = compute_utility(bundle, x_i, delta, Q_dense, lambda_, alpha, errors_i)
        return bundle, util, gap, runtime
    else:
        return np.zeros(M, dtype=bool), float('-inf'), gap, runtime


def twosls_smoke_test(delta_star, phi, z, beta_star):
    """Run 2SLS on delta_star = const + phi @ beta + xi, using z as instruments.

    Includes a constant since delta is demeaned but phi is not.
    Returns (beta_hat, max_abs_error).
    """
    M = phi.shape[0]
    ones = np.ones((M, 1))

    # Augment with constant: [1, phi] instrumented by [1, z]
    z_aug = np.hstack([ones, z])
    phi_aug = np.hstack([ones, phi])

    # First stage: phi_aug = z_aug @ Pi + residual
    ZtZ_inv = np.linalg.inv(z_aug.T @ z_aug)
    Pi_hat = ZtZ_inv @ z_aug.T @ phi_aug
    phi_hat = z_aug @ Pi_hat

    # Second stage: delta = phi_hat @ [const, beta] + error
    coefs = np.linalg.lstsq(phi_hat, delta_star, rcond=None)[0]
    beta_hat = coefs[1:]  # skip constant
    max_err = float(np.abs(beta_hat - beta_star).max())

    return beta_hat, max_err
