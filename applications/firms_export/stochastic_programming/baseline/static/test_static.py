#!/bin/env python
import sys
from pathlib import Path
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solver import TwoStageSolver
from oracles import build_oracles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "data"))
from prepare_data import main as load_data

COUNTRY = "MEX"
KEEP_TOP = 20
END_BUFFER = 3
N_SAMPLE = 200
N_SIMULATIONS = 1

BETA = 0.0
R = 1
SIGMA_1 = 1.0
SIGMA_2 = 1.0

thetas = [
    np.zeros(4),
    np.array([0.5, -5.0, -1.0, 0.1]),
    np.array([0.1, -1.0, -0.5, 0.0]),
    np.array([ 1.47413693 ,-2.89827683 ,-0.01765633 , 0.07047045]),
]
error_seeds = [42, 43]
names = ["rev", "entry_c", "entry_dist", "entry_syn_d"]

ctx = load_data(COUNTRY, KEEP_TOP, end_buffer=END_BUFFER,
                n_sample=N_SAMPLE)
M = ctx["M"]
n_obs = ctx["n_obs"]
n_cov = 4


def build_combined_input(ctx, R):
    rev_1 = ctx["rev_chars_1"][:, 0, :]
    state = ctx["state_chars"]
    entry_chars = ctx["entry_chars"]
    C = ctx["syn_chars"]
    switch = 1 - state
    syn = state @ C

    modular = np.stack([
        rev_1,
        switch,
        switch * entry_chars[None, :],
        switch * syn * entry_chars[None, :],
    ], axis=-1)

    return {
        "id_data": {
            "state_chars":   ctx["state_chars"],
            "rev_chars_1":   ctx["rev_chars_1"],
            "rev_chars_2":   ctx["rev_chars_2"],
            "obs_bundles":   ctx["obs_bundles"],
            "obs_bundles_2": ctx["obs_bundles_2"],
            "modular":          modular,
            "constraint_mask":  None,
        },
        "item_data": {
            "syn_chars":   ctx["syn_chars"],
            "entry_chars": ctx["entry_chars"],
            "beta":        BETA,
            "R":           R,
        },
    }


model = ce.Model()
is_root = model.is_root()

cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov,
                   "n_simulations": N_SIMULATIONS},
    "subproblem": {"gurobi_params": {"TimeLimit": 5}},
}
model.load_config(cfg)

if is_root:
    input_data = build_combined_input(ctx, R)
else:
    input_data = None
model.data.load_and_distribute_input_data(input_data)

model.data_manager.id_data = model.data_manager.local_data.id_data


def setup_mincut_errors(model, seed):
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    eps_1 = np.zeros((n, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps_1[i] = np.random.default_rng((seed, gid, 0)).normal(0, SIGMA_1, M)
    model.features.local_modular_errors = eps_1
    model.features._error_oracle = lambda bundles, ids: (eps_1[ids] * bundles).sum(-1)
    model.features._error_oracle_takes_data = False


for theta in thetas:
    if is_root:
        t_str = np.array2string(theta, precision=2, separator=", ")
        print(f"\n  theta = {t_str}")
        header = (f"  {'seed':>6}  {'method':>8}  {'f':>12}  "
                  + "  ".join(f"{n:>10}" for n in names)
                  + f"  {'|g|':>10}")
        print(header)
        print("  " + "-" * (len(header) - 2))

    for seed in error_seeds:
        setup_mincut_errors(model, seed)
        model.features.build_quadratic_covariates_from_data()
        model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
        model.subproblems.initialize_solver()

        f_mc, g_mc = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)

        if is_root:
            g_str = "  ".join(f"{g_mc[j]:+10.4f}" for j in range(n_cov))
            print(f"  {seed:>6}  {'mincut':>8}  {f_mc:12.6f}  {g_str}  {np.linalg.norm(g_mc):10.4f}")

        cov_oracle, err_oracle = build_oracles(
            model, seed=seed, sigma_1=SIGMA_1,
            sigma_2=SIGMA_2)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_oracle)
        model.features.set_error_oracle(err_oracle)

        f_sp, g_sp = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)

        if is_root:
            g_str = "  ".join(f"{g_sp[j]:+10.4f}" for j in range(n_cov))
            print(f"  {seed:>6}  {'SP(b=0)':>8}  {f_sp:12.6f}  {g_str}  {np.linalg.norm(g_sp):10.4f}")

            f_ok = np.isclose(f_mc, f_sp, atol=1e-6)
            g_ok = np.allclose(g_mc, g_sp, atol=1e-6)
            if f_ok and g_ok:
                print(f"  *** MATCH OK")
            else:
                parts = []
                if not f_ok:
                    parts.append(f"|df|={abs(f_mc - f_sp):.2e}")
                if not g_ok:
                    parts.append(f"max|dg|={np.max(np.abs(g_mc - g_sp)):.2e}")
                print(f"  *** MISMATCH: {', '.join(parts)}")
