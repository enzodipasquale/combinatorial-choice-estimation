#!/usr/bin/env python3
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from mpi4py import MPI

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.unit_demand.dgp import simulate_logit_individual
from paper.numerical_experiments.unit_demand.logit import estimate_logit_mle_individual


def choices_to_bundles(choices, J):
    N = len(choices)
    bundles = np.zeros((N, J), dtype=bool)
    inside = choices > 0
    bundles[inside, choices[inside] - 1] = True
    return bundles


def run_replication(N, J, K, beta, replication=0, config=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = config or {}
    exp = cfg.get("experiment", {})
    sigma = exp.get("sigma", 1.0)
    n_simulations = exp.get("n_simulations", 1)
    if n_simulations == "match_N":
        n_simulations = N
    rho = exp.get("covariate_correlation", 0.0)
    seed = replication * 1000 + 42

    # --- Data generation and MLE on rank 0 only ---
    if rank == 0:
        X, choices = simulate_logit_individual(
            N, J, K, beta, sigma=sigma, rho=rho, seed=seed)

        t0 = time.perf_counter()
        try:
            beta_mle, mle_result = estimate_logit_mle_individual(X, choices)
        except Exception:
            beta_mle = np.full(K, np.nan)
            mle_result = None
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
    model = ce.Model()
    model.load_config({
        "dimensions": {"n_obs": N, "n_items": J, "n_covariates": K,
                        "n_simulations": n_simulations},
        "subproblem": {"name": "UnitDemand", **cfg.get("subproblem", {})},
        "row_generation": cfg.get("row_generation", {}),
    })
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    model.features.build_local_modular_error_oracle(
        seed=3 * replication + 2, sigma=sigma, distribution='gumbel')
    model.subproblems.load_solver()

    t0 = time.perf_counter()
    result = model.point_estimation.one_slack.solve(verbose=False)
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
