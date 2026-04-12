"""Step 5 verification: custom find_best_item matches standalone greedy + speed test."""

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


def _greedy_with_find_best_item(phi, theta_mod, theta_gs, hubs_i, origin_of,
                                 errors_i, M, find_best_item_fn, local_id,
                                 data, modular_error):
    """Run greedy using the find_best_item interface (standalone, no combest)."""
    theta = np.concatenate([theta_mod, [theta_gs]])
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


def _make_mock_data(phi, origin_of, hubs_list):
    """Create a mock data object mimicking combest LocalData."""
    data = SimpleNamespace()
    data.item_data = {'phi': phi, 'origin_of': origin_of}
    data.id_data = {'hubs': hubs_list}
    return data


@pytest.mark.parametrize("C", [2, 3])
def test_find_best_item_matches_greedy(C):
    """At small M, find_best_item greedy == standalone greedy."""
    rng_geo = np.random.default_rng(42)
    locations, dists, populations = build_geography(C, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")

    N_AIRLINES = 10
    N_THETAS = 10
    rng = np.random.default_rng(142)
    find_best_item_fn = make_find_best_item()

    for t in range(N_THETAS):
        theta_mod = rng.uniform(-2, 2, phi.shape[1])
        theta_gs = rng.uniform(0.1, 3.0)
        hubs_list = build_hubs(N_AIRLINES, C, np.random.default_rng(42 + t))
        errors = rng.normal(0, 1.0, (N_AIRLINES, M))
        data = _make_mock_data(phi, origin_of, hubs_list)

        for i in range(N_AIRLINES):
            b_std, v_std = greedy_demand(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
            b_fbi, v_fbi = _greedy_with_find_best_item(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M,
                find_best_item_fn, i, data, errors[i])

            assert abs(v_std - v_fbi) < TOL, (
                f"find_best_item utility mismatch! C={C}, theta={t}, airline={i}: "
                f"std={v_std:.8f}, fbi={v_fbi:.8f}"
            )


def test_find_best_item_speed():
    """At M=110, custom find_best_item should run in reasonable time."""
    C = 11  # M = 11*10 = 110
    N = 5
    rng_geo = np.random.default_rng(77)
    locations, dists, populations = build_geography(C, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")
    hubs_list = build_hubs(N, C, np.random.default_rng(77))
    errors = rng_geo.normal(0, 1.0, (N, M))

    theta_mod = np.array([1.5, -0.5])
    theta_gs = 1.5
    find_best_item_fn = make_find_best_item()
    data = _make_mock_data(phi, origin_of, hubs_list)

    # Time the standalone greedy (naive)
    t0 = time.perf_counter()
    for i in range(N):
        greedy_demand(phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
    t_naive = time.perf_counter() - t0

    # Time find_best_item greedy
    t0 = time.perf_counter()
    for i in range(N):
        _greedy_with_find_best_item(
            phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M,
            find_best_item_fn, i, data, errors[i])
    t_fbi = time.perf_counter() - t0

    print(f"\nM={M}, N={N}: naive={t_naive:.3f}s, find_best_item={t_fbi:.3f}s, "
          f"ratio={t_naive/t_fbi:.1f}x")

    assert t_fbi < 30, f"find_best_item too slow: {t_fbi:.1f}s"
