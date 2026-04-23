#!/usr/bin/env python3
"""Single replication of the unit-demand efficiency benchmark.

Decouples the data-generating process (`dgp`) from the estimator assumption
(`est`).  Both MLE and combest's error oracle are built from `est`.
Supports: probit (iid normal), probit_corr (equicorrelated normal), logit.
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from numerical_experiments.unit_demand.dgp import (
    simulate_probit_individual, simulate_logit_individual)
from numerical_experiments.unit_demand.ghk import (
    estimate_probit_mle_individual)
from numerical_experiments.unit_demand.logit import (
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


def _run_probit_mle_with_sigma(X, choices, beta, Sigma, ghk_draws, seed):
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


def _probit_sigma(J, sigma, error_rho):
    """Equicorrelated covariance σ²·[(1−ρ)I + ρ·11ᵀ]; ρ=0 reduces to σ²I."""
    base = (1 - error_rho) * np.eye(J) + error_rho * np.ones((J, J))
    return sigma**2 * base


def run_replication(N, J, K, beta, replication=0, config=None,
                    dgp="probit", est=None, solver="one_slack"):
    """Run one replication.

    Parameters
    ----------
    dgp : {"probit", "probit_corr", "logit"}
        Data-generating process.  "probit_corr" uses equicorrelated normal
        errors with correlation `error_rho` from config (default 0.3).
    est : {"probit", "probit_corr", "logit"} or None
        Estimator assumption (MLE + combest oracle).
        Defaults to `dgp` (correctly-specified case).
    """
    if est is None:
        est = dgp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = config or {}
    exp = cfg.get("experiment", {})
    sigma = exp.get("sigma", 1.0)
    rho = exp.get("covariate_correlation", 0.0)
    error_rho = exp.get("error_correlation", 0.3)
    n_simulations = resolve_n_simulations(exp.get("n_simulations", 1), N)
    ghk_draws = exp.get("ghk_draws", 200)
    seed = replication * 1000 + 42

    # --- Data generation and MLE on rank 0 only ---
    if rank == 0:
        if dgp == "probit":
            X, choices = simulate_probit_individual(
                N, J, K, beta, Sigma=_probit_sigma(J, sigma, 0.0),
                rho=rho, seed=seed)
        elif dgp == "probit_corr":
            X, choices = simulate_probit_individual(
                N, J, K, beta, Sigma=_probit_sigma(J, sigma, error_rho),
                rho=rho, seed=seed)
        elif dgp == "logit":
            X, choices = simulate_logit_individual(
                N, J, K, beta, sigma=sigma, rho=rho, seed=seed)
        else:
            raise ValueError(f"Unknown dgp: {dgp}")

        t0 = time.perf_counter()
        if est == "probit":
            beta_mle, mle_result = _run_probit_mle_with_sigma(
                X, choices, beta, _probit_sigma(J, sigma, 0.0),
                ghk_draws, seed=seed + 1)
        elif est == "probit_corr":
            beta_mle, mle_result = _run_probit_mle_with_sigma(
                X, choices, beta, _probit_sigma(J, sigma, error_rho),
                ghk_draws, seed=seed + 1)
        elif est == "logit":
            beta_mle, mle_result = _run_logit_mle(X, choices)
        else:
            raise ValueError(f"Unknown est: {est}")
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
    if est == "logit":
        error_distribution = "gumbel"
        cov_mat = None
    elif est == "probit_corr":
        error_distribution = "normal"
        # sigma scaling is already in cov matrix; oracle applies Cholesky
        cov_mat = _probit_sigma(J, 1.0, error_rho)
    else:  # probit
        error_distribution = "normal"
        cov_mat = None
    cb.features.build_local_modular_error_oracle(
        seed=3 * replication + 2, sigma=sigma,
        distribution=error_distribution,
        covariance_matrix=cov_mat)
    cb.subproblems.load_solver()

    if solver not in {"one_slack", "n_slack"}:
        raise ValueError(f"Unknown solver: {solver}")
    t0 = time.perf_counter()
    result = getattr(cb.point_estimation, solver).solve(verbose=False)
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
