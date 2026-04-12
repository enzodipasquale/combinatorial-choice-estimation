"""Greedy must match brute-force at small M.

Mandatory test — 10 random airlines x 10 random thetas x random errors.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import build_geography, build_edges, build_covariates, build_hubs
from oracle import brute_force_demand, greedy_demand

N_AIRLINES = 10
N_THETAS = 10
TOL = 1e-8
POP_LOG_STD = 1.0


def _run_greedy_vs_brute(C, seed):
    rng_geo = np.random.default_rng(seed)
    locations, dists, populations = build_geography(C, POP_LOG_STD, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")

    rng_test = np.random.default_rng(seed + 100)

    for t in range(N_THETAS):
        theta_mod = np.abs(rng_test.uniform(0.1, 2, phi.shape[1]))
        theta_gs = rng_test.uniform(0.1, 3.0)
        hubs_list = build_hubs(N_AIRLINES, C, populations, 1, 3, 0.5,
                               np.random.default_rng(seed + t))
        errors = rng_test.normal(0, 1.0, (N_AIRLINES, M))

        for i in range(N_AIRLINES):
            b_greedy, v_greedy = greedy_demand(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
            b_brute, v_brute = brute_force_demand(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
            assert v_greedy >= v_brute - TOL, (
                f"Greedy < brute-force! C={C}, theta={t}, airline={i}: "
                f"greedy={v_greedy:.8f}, brute={v_brute:.8f}"
            )


@pytest.mark.parametrize("C", [2, 3])
def test_greedy_matches_brute_force(C):
    _run_greedy_vs_brute(C, seed=42)


def test_greedy_matches_brute_force_origin_fe():
    C = 3
    rng_geo = np.random.default_rng(99)
    locations, dists, populations = build_geography(C, POP_LOG_STD, rng_geo)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "origin")

    rng_test = np.random.default_rng(199)

    for t in range(N_THETAS):
        n_mod = phi.shape[1]
        theta_mod = np.abs(rng_test.uniform(0.1, 2, n_mod))
        theta_gs = rng_test.uniform(0.1, 3.0)
        hubs_list = build_hubs(N_AIRLINES, C, populations, 1, 3, 0.5,
                               np.random.default_rng(99 + t))
        errors = rng_test.normal(0, 1.0, (N_AIRLINES, M))

        for i in range(N_AIRLINES):
            b_greedy, v_greedy = greedy_demand(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
            b_brute, v_brute = brute_force_demand(
                phi, theta_mod, theta_gs, hubs_list[i], origin_of, errors[i], M)
            assert v_greedy >= v_brute - TOL
