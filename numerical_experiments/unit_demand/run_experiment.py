#!/usr/bin/env python3
"""Single replication of the unit-demand efficiency benchmark.

Selects DGP, MLE, and combest error oracle based on `model` ∈ {probit, logit}.
"""
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI

# Suppress spurious BLAS RuntimeWarnings from some OpenBLAS builds.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*matmul.*")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.unit_demand.dgp import (
    simulate_probit_individual, simulate_logit_individual)
from paper.numerical_experiments.unit_demand.ghk import (
    estimate_probit_mle_individual)
from paper.numerical_experiments.unit_demand.logit import (
    estimate_logit_mle_individual)


def choices_to_bundles(choices, J):
    N = len(choices)
    bundles = np.zeros((N, J), dtype=bool)
    inside = choices > 0
    bundles[inside, choices[inside] - 1] = True
    return bundles


def resolve_n_simulations(raw, N):
    if raw == "match_N":
        return N
    if raw == "match_sqrt_N":
        return int(np.ceil(np.sqrt(N)))
    return int(raw)


def _run_probit_mle(X, choices, beta, sigma, ghk_draws, seed):
    Sigma = sigma**2 * np.eye(X.shape[1])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            beta_mle, res = estimate_probit_mle_individual(
                X, choices, Sigma, R=ghk_draws, seed=seed, beta0=beta)
        return beta_mle, res
    except Exception:
        return np.full(len(beta), np.nan), None


def _run_logit_mle(X, choices):
    try:
        return estimate_logit_mle_individual(X, choices)
    except Exception:
        return np.full(X.shape[2], np.nan), None


def run_replication(N, J, K, beta, replication=0, config=None, model="probit"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = config or {}
    exp = cfg.get("experiment", {})
    sigma = exp.get("sigma", 1.0)
    rho = exp.get("covariate_correlation", 0.0)
    n_simulations = resolve_n_simulations(exp.get("n_simulations", 1), N)
    ghk_draws = exp.get("ghk_draws", 200)
    seed = replication * 1000 + 42

    # --- Data generation and MLE on rank 0 only ---
    if rank == 0:
        if model == "probit":
            Sigma = sigma**2 * np.eye(J)
            X, choices = simulate_probit_individual(
                N, J, K, beta, Sigma=Sigma, rho=rho, seed=seed)
        elif model == "logit":
            X, choices = simulate_logit_individual(
                N, J, K, beta, sigma=sigma, rho=rho, seed=seed)
        else:
            raise ValueError(f"Unknown model: {model}")

        t0 = time.perf_counter()
        if model == "probit":
            beta_mle, mle_result = _run_probit_mle(
                X, choices, beta, sigma, ghk_draws, seed=seed + 1)
        else:
            beta_mle, mle_result = _run_logit_mle(X, choices)
        runtime_mle = time.perf_counter() - t0
        mle_converged = mle_result is not None and mle_result.success

        obs_bundles = choices_to_bundles(choices, J)
        input_data = {
            "id_data": {"modular": X, "obs_bundles": obs_bundles},
            "item_data": {},
        }
    else:
        input_data = None
        beta_mle = None
        runtime_mle = None

    # --- Combest (all ranks) ---
    cb = ce.Model()
    cb.load_config({
        "dimensions": {"n_obs": N, "n_items": J, "n_covariates": K,
                        "n_simulations": n_simulations},
        "subproblem": {"name": "UnitDemand", **cfg.get("subproblem", {})},
        "row_generation": cfg.get("row_generation", {}),
    })
    cb.data.load_and_distribute_input_data(input_data)
    cb.features.build_quadratic_covariates_from_data()
    error_distribution = "gumbel" if model == "logit" else "normal"
    cb.features.build_local_modular_error_oracle(
        seed=3 * replication + 2, sigma=sigma,
        distribution=error_distribution)
    cb.subproblems.load_solver()

    t0 = time.perf_counter()
    result = cb.point_estimation.one_slack.solve(verbose=False)
    runtime_combest = time.perf_counter() - t0

    if rank != 0:
        return None

    return {
        "beta_mle": beta_mle.tolist(),
        "beta_combest": result.theta_hat.tolist(),
        "runtime_mle": runtime_mle,
        "runtime_combest": runtime_combest,
        "mle_converged": mle_converged,
    }
