"""Custom find_best_item matches standalone greedy + speed test."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import build_geography, build_edges, build_covariates, build_hubs
from oracle import greedy_demand, make_find_best_item

TOL = 1e-8
POP_LOG_STD = 1.0


def _greedy_with_find_best_item(phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                                 origin_of, errors_i, M, find_best_item_fn,
                                 local_id, data, modular_error):
    """Run greedy using the find_best_item interface (standalone)."""
    n_shared = phi.shape[1]
    N_firms = data.item_data['N_firms']
    # Build full theta vector: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
    theta = np.zeros(n_shared + N_firms + 1)
    theta[:n_shared] = theta_rev
    theta[n_shared + local_id] = theta_fc_i
    theta[-1] = theta_gs

    bundle = np.zeros(M, dtype=bool)
    items_left = np.ones(M, dtype=bool)
    best_val = 0.0
    cache = {}

    while np.any(items_left):
        best_item, val = find_best_item_fn(
            local_id, bundle, items_left, theta, best_val, data, modular_error,
            cache=cache)
        if val <= best_val:
            break
        bundle[best_item] = True
        items_left[best_item] = False
        best_val = val

    return bundle, best_val


def _make_mock_data(phi, origin_of, hubs_list, N_firms):
    data = SimpleNamespace()
    data.item_data = {'phi': phi, 'origin_of': origin_of, 'N_firms': N_firms}
    data.id_data = {'hubs': hubs_list}
    return data


@pytest.mark.parametrize("C", [2, 3])
def test_find_best_item_matches_greedy(C):
    rng_geo = np.random.default_rng(42)
    locations, dists, populations = build_geography(C, POP_LOG_STD, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")

    N_AIRLINES = 10
    N_THETAS = 10
    rng = np.random.default_rng(142)
    find_best_item_fn = make_find_best_item()

    for t in range(N_THETAS):
        theta_rev = np.abs(rng.uniform(0.1, 2, phi.shape[1]))
        theta_gs = rng.uniform(0.1, 3.0)
        hubs_list = build_hubs(N_AIRLINES, C, populations, 1, 3, 0.5,
                               np.random.default_rng(42 + t))
        errors = rng.normal(0, 1.0, (N_AIRLINES, M))
        data = _make_mock_data(phi, origin_of, hubs_list, N_AIRLINES)

        for i in range(N_AIRLINES):
            theta_fc_i = rng.uniform(0.1, 2.0)
            b_std, v_std = greedy_demand(
                phi, theta_rev, theta_fc_i, theta_gs, hubs_list[i],
                origin_of, errors[i], M)
            b_fbi, v_fbi = _greedy_with_find_best_item(
                phi, theta_rev, theta_fc_i, theta_gs, hubs_list[i],
                origin_of, errors[i], M,
                find_best_item_fn, i, data, errors[i])

            assert abs(v_std - v_fbi) < TOL, (
                f"find_best_item utility mismatch! C={C}, theta={t}, airline={i}: "
                f"std={v_std:.8f}, fbi={v_fbi:.8f}"
            )


def test_find_best_item_speed():
    C = 11  # M = 110
    N = 5
    rng_geo = np.random.default_rng(77)
    locations, dists, populations = build_geography(C, POP_LOG_STD, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")
    hubs_list = build_hubs(N, C, populations, 1, 3, 0.5, np.random.default_rng(77))
    errors = rng_geo.normal(0, 1.0, (N, M))

    theta_rev = np.array([0.5])
    theta_fc_i = 0.5
    theta_gs = 1.5
    find_best_item_fn = make_find_best_item()
    data = _make_mock_data(phi, origin_of, hubs_list, N)

    t0 = time.perf_counter()
    for i in range(N):
        greedy_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_list[i],
                     origin_of, errors[i], M)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(N):
        _greedy_with_find_best_item(
            phi, theta_rev, theta_fc_i, theta_gs, hubs_list[i],
            origin_of, errors[i], M,
            find_best_item_fn, i, data, errors[i])
    t_fbi = time.perf_counter() - t0

    print(f"\nM={M}, N={N}: naive={t_naive:.3f}s, find_best_item={t_fbi:.3f}s")
    assert t_fbi < 30
