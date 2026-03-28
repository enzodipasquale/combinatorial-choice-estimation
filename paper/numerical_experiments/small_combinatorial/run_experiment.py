#!/usr/bin/env python3
"""Run a single replication: both simulated MLE and combest estimation."""
import sys
import copy
import time
import warnings
import numpy as np
from pathlib import Path
from mpi4py import MPI

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.combest_scenarios.generate_data import generate_data
from paper.numerical_experiments.small_combinatorial.mle import estimate_smle


def run_replication(spec, N, J, alpha=None, lambda_val=None, replication=0, config=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = config or {}
    exp = cfg.get("experiment", {})

    beta_star = exp.get("beta_star", "ones")
    if beta_star == "ones":
        beta_star = None

    sigma = exp.get("sigma", 1.0)
    mle_draws = exp.get("mle_draws", 10000)

    # --- Data generation (rank 0) ---
    if rank == 0:
        input_data, theta_star = generate_data(
            spec, N, J, alpha=alpha, lambda_val=lambda_val,
            rho=exp.get("rho", 0.5), beta_star=beta_star,
            seed=replication * 1000 + 42)

        id_d, item_d = input_data["id_data"], input_data["item_data"]
        n_covariates = sum(
            d[k].shape[-1] for d in (id_d, item_d)
            for k in ("modular", "quadratic") if k in d
        )
        subproblem_name = cfg.get("specifications", {}).get(spec, {}).get("subproblem", "Greedy")
        dim_cfg = {"n_obs": N, "n_items": J, "n_covariates": n_covariates, "n_simulations": 1}
    else:
        input_data = theta_star = dim_cfg = subproblem_name = None

    dim_cfg = comm.bcast(dim_cfg, root=0)
    subproblem_name = comm.bcast(subproblem_name, root=0)

    subproblem_cfg = {"name": subproblem_name}
    subproblem_cfg.update(cfg.get("subproblem", {}))

    rg_cfg = copy.deepcopy(cfg.get("row_generation", {}))

    # Lambda >= 0 bounds for supermodular / quadratic knapsack
    if spec in ("supermodular", "quadratic_knapsack"):
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

    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    model.features.build_local_modular_error_oracle(seed=3 * replication + 1, sigma=sigma)
    model.subproblems.load_solver()

    # GS find_best_item optimization
    if spec == "gross_substitutes":
        def _gs_find_best_item(local_id, bundle, items_left, theta, best_val,
                               local_data, modular_error):
            M = len(bundle)
            x_i = local_data.id_data["modular"][local_id, :, 0]
            m = theta[0] * x_i - theta[1:M + 1] + modular_error
            candidates = np.where(items_left)[0]
            best_j = candidates[np.argmax(m[candidates])]
            new_val = best_val + m[best_j] - 2 * theta[M + 1] * int(bundle.sum())
            return best_j, new_val
        model.subproblems.subproblem_solver.find_best_item = _gs_find_best_item

    # Generate observed bundles
    theta_star = comm.bcast(theta_star, root=0)
    model.subproblems.generate_obs_bundles(theta_star)

    # Extract observed bundles for MLE (rank 0 gathers all)
    local_obs = model.data.local_obs_bundles  # (n_local, J) boolean
    all_obs = comm.gather(local_obs, root=0)

    # Re-seed errors for estimation
    model.features.build_local_modular_error_oracle(seed=3 * replication + 2, sigma=sigma)

    # --- Combest estimation ---
    t0 = time.perf_counter()
    result = model.point_estimation.n_slack.solve(verbose=False)
    runtime_combest = time.perf_counter() - t0

    if rank != 0:
        return None

    theta_hat_cb = result.theta_hat.copy()
    obs_bundles = np.concatenate(all_obs, axis=0)  # (N, J)

    # --- MLE estimation (rank 0 only) ---
    t0 = time.perf_counter()
    try:
        theta_hat_mle, mle_result = estimate_smle(
            input_data, obs_bundles, sigma=sigma, R=mle_draws,
            seed=replication * 1000 + 99, theta0=theta_star.copy(),
            maxiter=5000)
        mle_converged = mle_result.success
    except Exception as e:
        print(f"  MLE failed: {e}", flush=True)
        theta_hat_mle = np.full_like(theta_star, np.nan)
        mle_converged = False
    runtime_mle = time.perf_counter() - t0

    meta = input_data["meta"]
    alpha_idx = meta.get("alpha_indices", [])
    delta_idx = meta.get("delta_indices", list(range(len(theta_star))))
    lambda_idx = meta.get("lambda_indices", [])

    rg_bounds = cfg.get("row_generation", {})
    bound_lb = float(rg_bounds.get("theta_bounds", {}).get("lb", -100))
    bound_ub = float(rg_bounds.get("theta_bounds", {}).get("ub", 100))
    at_bound = lambda v: (v <= bound_lb + 1.0) | (v >= bound_ub - 1.0)

    return {
        "theta_combest": theta_hat_cb.tolist(),
        "theta_mle": theta_hat_mle.tolist(),
        "theta_star": theta_star.tolist(),
        "runtime_combest": runtime_combest,
        "runtime_mle": runtime_mle,
        "mle_converged": bool(mle_converged),
        "n_at_bound": int(np.sum(at_bound(theta_hat_cb))),
        "alpha_indices": alpha_idx,
        "delta_indices": delta_idx,
        "lambda_indices": lambda_idx,
    }
