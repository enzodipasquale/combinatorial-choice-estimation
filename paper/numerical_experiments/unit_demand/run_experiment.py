#!/usr/bin/env python3
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.unit_demand.dgp import (
    simulate_logit_individual, simulate_probit_individual)
from paper.numerical_experiments.unit_demand.logit import estimate_logit_mle_individual
from paper.numerical_experiments.unit_demand.ghk import estimate_probit_mle_individual


def choices_to_bundles(choices, J):
    N = len(choices)
    bundles = np.zeros((N, J), dtype=bool)
    inside = choices > 0
    bundles[inside, choices[inside] - 1] = True
    return bundles


def set_gumbel_oracle(model, seed, sigma=1.0):
    n_local = model.comm_manager.num_local_agent
    n_items = model.config.dimensions.n_items
    errors = np.zeros((n_local, n_items))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        rng = np.random.default_rng((seed, gid))
        eps = rng.gumbel(size=n_items + 1)
        errors[i] = sigma * (eps[1:] - eps[0])
    model.features.local_modular_errors = errors
    model.features.set_error_oracle(lambda bundles, ids: (errors[ids] * bundles).sum(-1))


def run_replication(dgp, N, J, K, beta, replication=0, config=None):
    cfg = config or {}
    exp = cfg.get("experiment", {})
    n_simulations = exp.get("n_simulations", 1)
    base_sigma = exp.get("sigma", 1.0)
    sigma = base_sigma * np.sqrt(K)
    seed = replication * 1000 + 42

    if dgp == "logit":
        X, choices = simulate_logit_individual(N, J, K, beta, sigma=sigma, seed=seed)
    else:
        Sigma = sigma**2 * np.eye(J)
        X, choices = simulate_probit_individual(N, J, K, beta, Sigma=Sigma, seed=seed)

    t0 = time.perf_counter()
    if dgp == "logit":
        gamma_mle, _ = estimate_logit_mle_individual(X, choices)
        beta_mle = gamma_mle * sigma
    else:
        beta_mle, _ = estimate_probit_mle_individual(X, choices, Sigma, seed=seed + 1)
    runtime_mle = time.perf_counter() - t0

    obs_bundles = choices_to_bundles(choices, J)
    input_data = {
        "id_data": {"modular": X, "obs_bundles": obs_bundles},
        "item_data": {},
    }

    model = ce.Model()
    model.load_config({
        "dimensions": {"n_obs": N, "n_items": J, "n_covariates": K, "n_simulations": n_simulations},
        "subproblem": {"name": "UnitDemand", **cfg.get("subproblem", {})},
        "row_generation": cfg.get("row_generation", {}),
    })
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    if dgp == "logit":
        set_gumbel_oracle(model, seed=3 * replication + 2, sigma=sigma)
    else:
        model.features.build_local_modular_error_oracle(
            seed=3 * replication + 2, covariance_matrix=Sigma)

    model.subproblems.load_solver()

    t0 = time.perf_counter()
    result = model.point_estimation.n_slack.solve(verbose=False)
    runtime_combest = time.perf_counter() - t0

    return {
        "beta_star": beta.tolist(),
        "beta_mle": beta_mle.tolist(),
        "beta_combest": result.theta_hat.tolist(),
        "runtime_mle": runtime_mle,
        "runtime_combest": runtime_combest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", required=True, choices=["logit", "probit"])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--J", type=int, required=True)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(Path(__file__).parent / args.config) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    beta = np.array(exp["beta_star"])

    result = run_replication(args.dgp, args.N, args.J, exp["K"], beta,
                             replication=args.replication, config=config)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "results" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.dgp}_N{args.N}_J{args.J}_rep{args.replication}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
