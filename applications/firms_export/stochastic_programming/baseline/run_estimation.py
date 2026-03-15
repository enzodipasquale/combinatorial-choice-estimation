#!/bin/env python
import sys
from pathlib import Path
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_data_sp import main as load_data, build_input_data_sp

# ── Settings ──────────────────────────────────────────────────────────
COUNTRY = "MEX"
KEEP_TOP = 10
BETA = 0.8
END_BUFFER = 3
R = 100
N_SAMPLE = 500
SIGMA_EPS = 1.0
SIGMA_NU_1 = 0.5
SIGMA_NU_2 = 2
N_SIMULATIONS = 1

SEED = 42
MAX_ITERS = 50
TAU = 1.0


def build_model(n_sample=N_SAMPLE):
    model = ce.Model()

    if model.is_root():
        ctx = load_data(COUNTRY, KEEP_TOP, beta=BETA, end_buffer=END_BUFFER,
                        n_sample=n_sample)
        input_data = build_input_data_sp(ctx, R=R)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
    else:
        input_data, n_obs, M = None, None, None

    n_obs = model.comm_manager.bcast(n_obs)
    M = model.comm_manager.bcast(M)

    n_rev = 1
    n_cov = n_rev + 3  # rev, s, sc, c

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": n_cov, "n_simulations": N_SIMULATIONS},
        "subproblem": {"gurobi_params": {"TimeLimit": 10}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    cov_oracle, err_oracle = build_oracles(model, seed=SEED,
                                           sigma_eps=SIGMA_EPS,
                                           sigma_nu_1=SIGMA_NU_1,
                                           sigma_nu_2=SIGMA_NU_2)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(cov_oracle)
    model.features.set_error_oracle(err_oracle)

    return model


if __name__ == "__main__":
    model = build_model()
    is_root = model.comm_manager.is_root()

    # Starting point
    theta0 = np.array([0.01, -5.0, -1.0, 0.2])
    names = ["rev", "s", "sc", "c"]

    if is_root:
        print(f"\nStarting bundle method: theta0 = {theta0}")
        print(f"  R={R}, S={N_SIMULATIONS}, beta={BETA}, seed={SEED}")

    # Evaluate gradient at theta0 first
    f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta0)
    if is_root:
        g_str = "  ".join(f"{names[j]}={g_val[j]:+.4f}" for j in range(len(names)))
        print(f"f(theta0) = {f_val:.6f}  grad: {g_str}  |g|={np.linalg.norm(g_val):.4f}")

    # Run bundle method
    result = model.point_estimation.bundle.solve(
        theta0, tau=TAU, max_iters=MAX_ITERS, verbose=True)

    result = model.point_estimation.bundle.solve(
        result.theta_hat, tau=TAU, max_iters=MAX_ITERS, verbose=True)

    if is_root:
        print(f"\ntheta_hat = {result.theta_hat}")
        print(f"obj={result.final_objective:.6f}  iters={result.num_iterations}  "
              f"converged={result.converged}  time={result.total_time:.1f}s")

        f_hat, g_hat = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(
            result.theta_hat)
        print(f"f(theta_hat) = {f_hat:.6f}  |grad| = {np.linalg.norm(g_hat):.6f}")
