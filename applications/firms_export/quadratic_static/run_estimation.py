#!/bin/env python
from pathlib import Path
import yaml
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from oracles import build_oracles
from prepare_data import main as load_data

with open(Path(__file__).resolve().parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

COUNTRY = CFG["data"]["country"]
KEEP_TOP = CFG["data"]["keep_top"]
START_BUFFER = CFG["data"]["start_buffer"]
N_SAMPLE = CFG["data"]["n_sample"]

SEED = CFG["estimation"]["seed"]
SIGMA_PERM = CFG["estimation"]["sigma_perm"]
SIGMA_TRANS = CFG["estimation"]["sigma_trans"]
N_SIMULATIONS = CFG["estimation"]["n_simulations"]
MAX_RG_ITERS = CFG["estimation"]["max_rg_iters"]

N_COV = 5
NAMES = ["rev", "fc", "dist", "syn", "syn_d"]


if __name__ == "__main__":
    model = ce.Model()
    is_root = model.is_root()

    if is_root:
        ctx = load_data(COUNTRY, KEEP_TOP, start_buffer=START_BUFFER,
                        n_sample=N_SAMPLE)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
        firm_idx = ctx["firm_idx"]
        dist_home = ctx["dist_home"]
        syn_chars = ctx["syn_chars"]

        modular = np.stack([
            ctx["rev_chars"],
            -np.ones((n_obs, M)),
            -np.broadcast_to(dist_home[None, :], (n_obs, M)),
        ], axis=-1)

        C = syn_chars
        C_d = syn_chars * (dist_home[:, None] + dist_home[None, :]) / 2
        quadratic = np.stack([C, C_d], axis=-1)

        input_data = {
            "id_data": {
                "obs_bundles": ctx["obs_bundles"],
                "modular": modular,
                "constraint_mask": None,
                "firm_idx": firm_idx,
            },
            "item_data": {
                "quadratic": quadratic,
            },
        }

        print(f"Data: {COUNTRY}, top {KEEP_TOP} destinations")
        print(f"  N={n_obs}, M={M}")
        print(f"  unique firms: {len(np.unique(firm_idx))}")
        print(f"  bundle sizes: mean={ctx['obs_bundles'].sum(1).mean():.2f}, "
              f"max={ctx['obs_bundles'].sum(1).max()}, "
              f"zero={(~ctx['obs_bundles'].any(1)).mean():.1%}")
    else:
        input_data, n_obs, M, firm_idx = None, None, None, None

    n_obs = model.comm_manager.bcast(n_obs)
    M = model.comm_manager.bcast(M)
    firm_idx = model.comm_manager.bcast(firm_idx)

    cfg = {
        "dimensions": {"n_obs": n_obs, "n_items": M,
                       "n_covariates": N_COV, "n_simulations": N_SIMULATIONS},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": 0}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    err_oracle = build_oracles(
        model, firm_idx, seed=SEED,
        sigma_perm=SIGMA_PERM, sigma_trans=SIGMA_TRANS)

    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()
    model.features.build_quadratic_covariates_from_data()
    model.features.set_error_oracle(err_oracle)

    if is_root:
        print(f"\nEstimation: seed={SEED}, sigma_perm={SIGMA_PERM}, "
              f"sigma_trans={SIGMA_TRANS}")
        print(f"  n_simulations={N_SIMULATIONS}, max_rg_iters={MAX_RG_ITERS}")

    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, verbose=True)

    if is_root:
        print(f"\ntheta_hat = {result.theta_hat}")
        for j, name in enumerate(NAMES):
            print(f"  {name:>8} = {result.theta_hat[j]:+.6f}")
        print(f"obj={result.final_objective:.6f}  "
              f"iters={result.num_iterations}  time={result.total_time:.1f}s")
