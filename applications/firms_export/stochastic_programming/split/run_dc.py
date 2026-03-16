#!/bin/env python
import sys
from pathlib import Path
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from solver import TwoStageSolverSplit
from oracles import build_oracles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from dc import DCSolver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
from prepare_data import main as load_data, build_input_data

COUNTRY = "MEX"
KEEP_TOP = 20
END_BUFFER = 2
N_SAMPLE = 5000
N_SIMULATIONS = 1

R = 200
SIGMA_1 = 1.0
SIGMA_2 = 1.0

SEED = 42
MAX_DC_ITERS = 20
MAX_RG_ITERS = 200
DC_TOL = 1e-6

N_COV_STATIC = 4
N_COV = 8
NAMES = ["rev_1", "entry_c_1", "entry_dist_1", "syn_1",
         "rev_2", "entry_c_2", "entry_dist_2", "syn_2"]


def load_ctx():
    model = ce.Model()
    if model.is_root():
        ctx = load_data(COUNTRY, KEEP_TOP, end_buffer=END_BUFFER,
                        n_sample=N_SAMPLE)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
    else:
        ctx, n_obs, M = None, None, None
    n_obs = model.comm_manager.bcast(n_obs)
    M = model.comm_manager.bcast(M)
    return model, ctx, n_obs, M


def build_static_model(base_model, ctx, n_obs, M):
    m = ce.Model()

    if m.is_root():
        state_chars = ctx["state_chars"]
        entry_chars = ctx["entry_chars"]
        syn_chars = ctx["syn_chars"]
        switch = 1 - state_chars
        syn_state = state_chars @ syn_chars

        modular = np.stack([
            ctx["rev_chars_1"][:, 0, :],
            switch,
            switch * entry_chars[None, :],
            switch * syn_state * entry_chars[None, :],
        ], axis=-1)

        input_data = {
            "id_data": {"obs_bundles": ctx["obs_bundles"],
                        "modular": modular, "constraint_mask": None},
            "item_data": {},
        }
    else:
        input_data = None

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": N_COV_STATIC, "n_simulations": 1},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": -1000}},
    }
    m.load_config(cfg)
    m.data.load_and_distribute_input_data(input_data)

    n_local = m.comm_manager.num_local_agent
    eps_1 = np.zeros((n_local, M))
    for i, gid in enumerate(m.comm_manager.agent_ids):
        eps_1[i] = np.random.default_rng((SEED, gid, 0)).normal(0, SIGMA_1, M)
    m.features.local_modular_errors = eps_1
    m.features._error_oracle = lambda bundles, ids: (eps_1[ids] * bundles).sum(-1)
    m.features._error_oracle_takes_data = False
    m.data_manager.id_data = m.data_manager.local_data.id_data
    m.features.build_quadratic_covariates_from_data()

    m.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    m.subproblems.initialize_solver()
    return m


def build_dc_model(ctx, n_obs, M):
    m = ce.Model()

    if m.is_root():
        input_data = build_input_data(ctx, R=R)
    else:
        input_data = None

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": N_COV, "n_simulations": N_SIMULATIONS},
        "subproblem": {"gurobi_params": {"TimeLimit": 10}},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": -1000}},
    }
    m.load_config(cfg)
    m.data.load_and_distribute_input_data(input_data)

    cov_oracle, err_oracle = build_oracles(m, seed=SEED,
                                           sigma_1=SIGMA_1,
                                           sigma_2=SIGMA_2)
    m.subproblems.load_solver(TwoStageSolverSplit)
    m.subproblems.initialize_solver()
    m.features.set_covariates_oracle(cov_oracle)
    m.features.set_error_oracle(err_oracle)
    return m


if __name__ == "__main__":
    base_model, ctx, n_obs, M = load_ctx()
    is_root = base_model.comm_manager.is_root()

    if is_root:
        print(f"\n{'='*60}")
        print(f"  1. STATIC (MinCut)")
        print(f"{'='*60}")

    model_mc = build_static_model(base_model, ctx, n_obs, M)
    result_mc = model_mc.point_estimation.n_slack.solve(
        initialize_solver=False, verbose=True)

    theta0 = np.empty(N_COV, dtype=np.float64)
    if is_root:
        th = result_mc.theta_hat
        theta0[:N_COV_STATIC] = th
        theta0[N_COV_STATIC:] = th
        print(f"\nStatic theta_hat = {th}")
        for j, name in enumerate(NAMES[:N_COV_STATIC]):
            print(f"  {name:>12} = {th[j]:+.6f}")
        print(f"obj={result_mc.final_objective:.6f}  "
              f"iters={result_mc.num_iterations}  time={result_mc.total_time:.1f}s")
    theta0 = base_model.comm_manager.Bcast(theta0)

    if is_root:
        print(f"\n{'='*60}")
        print(f"  2. DC (split, R={R})")
        print(f"{'='*60}")
        print(f"  theta0 (from static, duplicated) = {theta0}")
        print(f"  N={N_SAMPLE}, M={M}, seed={SEED}")
        print(f"  sigma_1={SIGMA_1}, sigma_2={SIGMA_2}")

    model_dc = build_dc_model(ctx, n_obs, M)
    solver = model_dc.subproblems.subproblem_solver
    row_gen = model_dc.point_estimation.n_slack
    dc = DCSolver(row_gen, solver)

    result = dc.solve(theta0, max_dc_iters=MAX_DC_ITERS, tol=DC_TOL,
                      verbose=True)

    if is_root and result is not None:
        print(f"\nDC theta_hat = {result.theta_hat}")
        for j, name in enumerate(NAMES):
            print(f"  {name:>12} = {result.theta_hat[j]:+.6f}")
        print(f"obj={result.final_objective:.6f}  dc_iters={result.num_iterations}  "
              f"converged={result.converged}  time={result.total_time:.1f}s")
