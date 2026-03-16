#!/bin/env python
import sys
from pathlib import Path
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles
from dc import DCSolver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
from prepare_data import main as load_data, build_input_data

COUNTRY = "MEX"
KEEP_TOP = 20
END_BUFFER = 3
N_SAMPLE = 200
N_SIMULATIONS = 1

BETA = 0.0
R = 1
SIGMA_1 = 1.0
SIGMA_2 = 1.0

SEED = 42
MAX_DC_ITERS = 20
MAX_RG_ITERS = 200
DC_TOL = 1e-6


def build_model(n_sample=N_SAMPLE):
    model = ce.Model()

    if model.is_root():
        ctx = load_data(COUNTRY, KEEP_TOP, end_buffer=END_BUFFER,
                        n_sample=n_sample)
        input_data = build_input_data(ctx, R=R, beta=BETA)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
    else:
        input_data, n_obs, M = None, None, None

    n_obs = model.comm_manager.bcast(n_obs)
    M = model.comm_manager.bcast(M)

    n_cov = 4
    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": n_cov, "n_simulations": N_SIMULATIONS},
        "subproblem": {"gurobi_params": {"TimeLimit": 10}},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": -1000}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    cov_oracle, err_oracle = build_oracles(model, seed=SEED,
                                           sigma_1=SIGMA_1,
                                           sigma_2=SIGMA_2)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(cov_oracle)
    model.features.set_error_oracle(err_oracle)

    return model


if __name__ == "__main__":
    model = build_model()
    is_root = model.comm_manager.is_root()

    theta0 = np.array([1.47413693, -2.89827683, -0.01765633, 0.07047045])
    names = ["rev", "entry_c", "entry_dist", "entry_syn_d"]

    if is_root:
        print(f"\nStarting DC algorithm (static, beta=0): theta0 = {theta0}")
        print(f"  R={R}, beta={BETA}, seed={SEED}")
        print(f"  N={N_SAMPLE}, M={model.n_items}")
        print(f"  sigma_1={SIGMA_1}, sigma_2={SIGMA_2}")

    solver = model.subproblems.subproblem_solver
    row_gen = model.point_estimation.n_slack
    dc = DCSolver(row_gen, solver)

    result = dc.solve(theta0, max_dc_iters=MAX_DC_ITERS, tol=DC_TOL,
                      verbose=True)

    if is_root and result is not None:
        print(f"\ntheta_hat = {result.theta_hat}")
        for j, name in enumerate(names):
            print(f"  {name:>12} = {result.theta_hat[j]:+.6f}")
        print(f"obj={result.final_objective:.6f}  dc_iters={result.num_iterations}  "
              f"converged={result.converged}  time={result.total_time:.1f}s")
