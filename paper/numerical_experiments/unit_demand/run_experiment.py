#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.unit_demand.dgp import simulate_probit_individual
from paper.numerical_experiments.unit_demand.ghk import estimate_probit_mle_individual


def choices_to_bundles(choices, J):
    N = len(choices)
    bundles = np.zeros((N, J), dtype=bool)
    inside = choices > 0
    bundles[inside, choices[inside] - 1] = True
    return bundles


def run_replication(N, J, K, beta, replication=0, config=None):
    cfg = config or {}
    exp = cfg.get("experiment", {})
    sigma = exp.get("sigma", 1.0)
    n_simulations = exp.get("n_simulations", 1)
    ghk_draws = exp.get("ghk_draws", 200)
    rho = exp.get("covariate_correlation", 0.0)
    seed = replication * 1000 + 42

    Sigma = sigma**2 * np.eye(J)
    X, choices = simulate_probit_individual(N, J, K, beta, Sigma=Sigma, rho=rho, seed=seed)

    # --- MLE ---
    t0 = time.perf_counter()
    beta_mle, _ = estimate_probit_mle_individual(
        X, choices, Sigma, R=ghk_draws, seed=seed + 1)
    runtime_mle = time.perf_counter() - t0

    # --- Combest ---
    obs_bundles = choices_to_bundles(choices, J)
    input_data = {
        "id_data": {"modular": X, "obs_bundles": obs_bundles},
        "item_data": {},
    }
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
        seed=3 * replication + 2, covariance_matrix=Sigma)
    model.subproblems.load_solver()

    t0 = time.perf_counter()
    result = model.point_estimation.n_slack.solve(verbose=False)
    runtime_combest = time.perf_counter() - t0

    return {
        "beta_mle": beta_mle.tolist(),
        "beta_combest": result.theta_hat.tolist(),
        "runtime_mle": runtime_mle,
        "runtime_combest": runtime_combest,
    }
