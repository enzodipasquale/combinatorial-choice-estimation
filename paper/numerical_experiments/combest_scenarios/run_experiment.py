#!/usr/bin/env python3
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.combest_scenarios.generate_data import generate_data


def run_replication(spec, N, M, alpha=None, lambda_val=None, replication=0, config=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = config or {}
    exp = cfg.get("experiment", {})

    beta_star = exp.get("beta_star", "ones")
    if beta_star == "ones":
        beta_star = None

    if rank == 0:
        input_data, theta_star = generate_data(
            spec, N, M, alpha=alpha, lambda_val=lambda_val,
            rho=exp.get("rho", 0.5), beta_star=beta_star,
            seed=replication * 1000 + 42)

        id_d, item_d = input_data["id_data"], input_data["item_data"]
        n_covariates = sum(
            d[k].shape[-1] for d in (id_d, item_d)
            for k in ("modular", "quadratic") if k in d
        )
        subproblem_name = cfg.get("specifications", {}).get(spec, {}).get("subproblem", "Greedy")
        dim_cfg = {"n_obs": N, "n_items": M, "n_covariates": n_covariates, "n_simulations": 1}
    else:
        input_data = theta_star = dim_cfg = subproblem_name = None

    dim_cfg = comm.bcast(dim_cfg, root=0)
    subproblem_name = comm.bcast(subproblem_name, root=0)

    subproblem_cfg = {"name": subproblem_name}
    subproblem_cfg.update(cfg.get("subproblem", {}))

    rg_cfg = dict(cfg.get("row_generation", {}))

    # Supermodular / quadratic knapsack require lambda >= 0 for the subproblem solver.
    # Enforce lb=0 on the quadratic (lambda) covariate indices.
    if spec in ("supermodular", "quadratic_knapsack"):
        n_cov = dim_cfg["n_covariates"]
        lambda_indices = comm.bcast(
            input_data["meta"]["lambda_indices"] if rank == 0 else None, root=0)
        lbs = dict(rg_cfg.get("theta_bounds", {}).get("lbs", {}))
        for idx in lambda_indices:
            lbs[str(idx)] = 0.0
        rg_cfg.setdefault("theta_bounds", {})["lbs"] = lbs

    model = ce.Model()
    model.load_config({
        "dimensions": dim_cfg,
        "subproblem": subproblem_cfg,
        "row_generation": rg_cfg,
    })
    sigma = exp.get("sigma", 1.0)

    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    model.features.build_local_modular_error_oracle(seed=3 * replication + 1, sigma=sigma)
    model.subproblems.load_solver()

    theta_star = comm.bcast(theta_star, root=0)
    model.subproblems.generate_obs_bundles(theta_star)
    model.features.build_local_modular_error_oracle(seed=3 * replication + 2, sigma=sigma)

    # Gurobi timeout callback (speeds up MIP subproblems like QuadraticKnapsackGRB)
    iteration_callback = None
    callbacks_cfg = cfg.get("callbacks", {})
    if callbacks_cfg.get("row_gen"):
        from combest.estimation.callbacks import adaptive_gurobi_timeout
        iteration_callback, _ = adaptive_gurobi_timeout(callbacks_cfg["row_gen"])

    t0 = time.perf_counter()
    result = model.point_estimation.n_slack.solve(
        iteration_callback=iteration_callback, verbose=False)
    runtime = time.perf_counter() - t0

    if rank != 0:
        return None

    theta_hat = result.theta_hat.copy()

    meta = input_data["meta"]
    alpha_idx = np.array(meta.get("alpha_indices", []))
    lambda_idx = np.array(meta.get("lambda_indices", []))
    delta_idx = np.array(meta.get("delta_indices", list(range(len(theta_star)))))

    rg_cfg = cfg.get("row_generation", {})
    bound_lb = float(rg_cfg.get("theta_bounds", {}).get("lb", -100))
    bound_ub = float(rg_cfg.get("theta_bounds", {}).get("ub", 100))
    at_bound = lambda v: (v <= bound_lb + 1.0) | (v >= bound_ub - 1.0)

    return {
        "theta_hat": theta_hat.tolist(),
        "theta_star": theta_star.tolist(),
        "runtime": runtime,
        "n_at_bound": int(np.sum(at_bound(theta_hat))),
        "alpha_indices": alpha_idx.tolist(),
        "lambda_indices": lambda_idx.tolist(),
        "delta_indices": delta_idx.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True,
                        choices=["gross_substitutes", "supermodular", "linear_knapsack", "quadratic_knapsack"])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=None)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(Path(__file__).parent / args.config) as f:
        config = yaml.safe_load(f)

    result = run_replication(
        args.spec, args.N, args.M, alpha=args.alpha,
        lambda_val=args.lambda_val, replication=args.replication, config=config)

    if result is not None:
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent / "results" / "raw"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.spec}_N{args.N}_M{args.M}_rep{args.replication}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
