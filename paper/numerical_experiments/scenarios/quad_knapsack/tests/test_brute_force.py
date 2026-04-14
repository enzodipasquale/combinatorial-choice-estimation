"""Verify QKP Gurobi solver matches brute-force at tiny size (M=15)."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import (build_items, build_adjacency, build_weights_capacities,
                           build_modular_features, build_blp_data, solve_all_agents)
from oracle import brute_force_demand, compute_utility, solve_qkp_gurobi


@pytest.fixture(scope='module')
def tiny_dgp():
    """Generate a tiny DGP (M=15, N=10) for brute-force comparison."""
    M, N = 15, 10
    rng = np.random.default_rng(42)

    locations, dists = build_items(M, rng)
    Q_sparse, Q_dense, avg_deg, nnz = build_adjacency(M, dists, 8, 15)

    dgp_cfg = {'weight_lo': 0.5, 'weight_hi': 1.5,
               'capacity_frac_lo': 0.3, 'capacity_frac_hi': 0.5}
    weights, capacities = build_weights_capacities(M, N, dgp_cfg, rng)
    x = build_modular_features(N, M, rng)

    rng_err = np.random.default_rng(1041)
    errors = rng_err.normal(0, 1.0, (N, M))

    alpha, lambda_ = 0.1, 0.05
    blp_cfg = {'K_phi': 3, 'rho': 0.5, 'delta_std': 0.5,
               'pi_0': 1.0, 'pi_z_std': 0.5, 'pi_xi': 0.3,
               'price_noise_std': 0.2}
    blp = build_blp_data(M, blp_cfg, np.random.default_rng(42))
    delta = blp['delta']

    return {
        'M': M, 'N': N, 'Q_dense': Q_dense, 'weights': weights,
        'capacities': capacities, 'x': x, 'errors': errors,
        'alpha': alpha, 'lambda_': lambda_, 'delta': delta,
    }


def test_qkp_matches_brute_force(tiny_dgp):
    """QKP solution must match brute-force for all agents at M=15."""
    d = tiny_dgp
    M, N = d['M'], d['N']

    for i in range(N):
        bf_bundle, bf_val = brute_force_demand(
            d['x'][i], d['delta'], d['Q_dense'], d['lambda_'], d['alpha'],
            d['errors'][i], d['weights'], d['capacities'][i], M)

        qkp_bundle, qkp_val, gap, _ = solve_qkp_gurobi(
            d['x'][i], d['delta'], d['Q_dense'], d['lambda_'], d['alpha'],
            d['errors'][i], d['weights'], d['capacities'][i], M,
            solver_cfg={'MIPGap': 0, 'OutputFlag': 0, 'TimeLimit': 10})

        assert gap == 0, f"Agent {i}: MIPGap={gap} (must be 0)"
        assert abs(qkp_val - bf_val) < 1e-6, \
            f"Agent {i}: QKP val={qkp_val:.8f} vs BF val={bf_val:.8f}"


def test_empty_bundle_feasible(tiny_dgp):
    """Empty bundle is always feasible (capacity satisfied trivially)."""
    d = tiny_dgp
    empty = np.zeros(d['M'], dtype=bool)
    val = compute_utility(empty, d['x'][0], d['delta'], d['Q_dense'],
                          d['lambda_'], d['alpha'], d['errors'][0])
    assert val == 0.0
