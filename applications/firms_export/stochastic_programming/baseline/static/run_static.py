#!/bin/env python
import sys
from pathlib import Path
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "data"))
from prepare_data import main as load_data

COUNTRY = "MEX"
KEEP_TOP = 50
END_BUFFER = 3
N_SAMPLE = 50000
N_SIMULATIONS = 1

SIGMA_EPS = 1.0
SIGMA_NU_1 = 1

SEED = 42
MAX_ITERS = 200


def build_model(n_sample=N_SAMPLE):
    model = ce.Model()

    if model.is_root():
        ctx = load_data(COUNTRY, KEEP_TOP, beta=0, end_buffer=END_BUFFER,
                        n_sample=n_sample)

        rev_1 = ctx["rev_chars_1"][:, 0, :]
        state = ctx["state_chars"]
        entry_chars = ctx["home_to_dest"]["dist"] / 1e3
        switch = 1 - state

        dist_pw = ctx["pairwise_features"]["dist"].values.astype(float) / 1e3
        C = np.exp(-dist_pw)
        np.fill_diagonal(C, 0.0)

        syn = state @ C

        modular = np.stack([
            rev_1,
            switch,
            switch * entry_chars[None, :],
            switch * syn * entry_chars[None, :],
        ], axis=-1)

        input_data = {
            "id_data": {
                "obs_bundles":      ctx["obs_bundles"],
                "modular":          modular,
                "constraint_mask":  None,
            },
            "item_data": {},
        }
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
        "row_generation": {
            "max_iters": MAX_ITERS,
            "theta_bounds": {
                "lb": -1000, "ub": 1000,
                "lbs": {"3": 0},
            },
        },
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    n_local = model.comm_manager.num_local_agent
    eps_1 = np.zeros((n_local, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps = np.random.default_rng((SEED, gid, 0)).normal(0, SIGMA_EPS, M)
        nu1 = np.random.default_rng((SEED, gid, 1)).normal(0, SIGMA_NU_1, M)
        eps_1[i] = eps + nu1
    model.features.local_modular_errors = eps_1
    model.features._error_oracle = lambda bundles, ids: (eps_1[ids] * bundles).sum(-1)
    model.features._error_oracle_takes_data = False

    model.data_manager.id_data = model.data_manager.local_data.id_data

    model.features.build_quadratic_covariates_from_data()

    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()

    return model


if __name__ == "__main__":
    model = build_model()
    is_root = model.comm_manager.is_root()
    names = ["rev", "entry_c", "entry_dist", "entry_syn_d"]

    if is_root:
        print(f"\nStarting row generation (static, beta=0)")
        print(f"  N={N_SAMPLE}, M={model.n_items}, seed={SEED}")

    result = model.row_generation.solve(verbose=True)

    if is_root:
        print(f"\ntheta_hat = {result.theta_hat}")
        for j, name in enumerate(names):
            print(f"  {name:>12} = {result.theta_hat[j]:+.6f}")
        print(f"obj={result.final_objective:.6f}  iters={result.num_iterations}  "
              f"converged={result.converged}  time={result.total_time:.1f}s")

    theta_hat = model.comm_manager.Bcast(
        result.theta_hat if is_root else np.zeros(len(names)))
    f_hat, g_hat = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_hat)
    if is_root:
        g_str = "  ".join(f"{names[j]}={g_hat[j]:+.4f}" for j in range(len(names)))
        print(f"f(theta_hat) = {f_hat:.6f}  grad: {g_str}  |g|={np.linalg.norm(g_hat):.6f}")
