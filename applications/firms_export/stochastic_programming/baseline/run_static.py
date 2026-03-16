#!/bin/env python
import sys
from pathlib import Path
import yaml
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
from prepare_data import main as load_data

with open(Path(__file__).resolve().parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

COUNTRY = CFG["data"]["country"]
KEEP_TOP = CFG["data"]["keep_top"]
END_BUFFER = CFG["data"]["end_buffer"]
N_SAMPLE = CFG["static"]["n_sample"]
SIGMA_1 = CFG["estimation"]["sigma_1"]
SEED = CFG["estimation"]["seed"]
MAX_RG_ITERS = CFG["estimation"]["max_rg_iters"]

N_COV = 4
NAMES = ["rev", "entry_c", "entry_dist", "entry_syn_d"]


if __name__ == "__main__":
    model = ce.Model()
    is_root = model.is_root()

    if is_root:
        ctx = load_data(COUNTRY, KEEP_TOP, end_buffer=END_BUFFER,
                        n_sample=N_SAMPLE)
        n_obs = ctx["n_obs"]
        M = ctx["M"]
    else:
        ctx, n_obs, M = None, None, None
    n_obs = model.comm_manager.bcast(n_obs)
    M = model.comm_manager.bcast(M)

    if is_root:
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
                       "n_covariates": N_COV, "n_simulations": 1},
        "row_generation": {"max_iters": MAX_RG_ITERS, "tolerance": 1e-6,
                          "theta_bounds": {"lb": -1000}},
    }
    model.load_config(cfg)
    model.data.load_and_distribute_input_data(input_data)

    n_local = model.comm_manager.num_local_agent
    eps_1 = np.zeros((n_local, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps_1[i] = np.random.default_rng((SEED, gid, 0)).normal(0, SIGMA_1, M)
    model.features.local_modular_errors = eps_1
    model.features._error_oracle = lambda bundles, ids: (eps_1[ids] * bundles).sum(-1)
    model.features._error_oracle_takes_data = False
    model.data_manager.id_data = model.data_manager.local_data.id_data
    model.features.build_quadratic_covariates_from_data()

    model.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    model.subproblems.initialize_solver()

    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, verbose=True)

    if is_root:
        print(f"\ntheta_hat = {result.theta_hat}")
        for j, name in enumerate(NAMES):
            print(f"  {name:>12} = {result.theta_hat[j]:+.6f}")
        print(f"obj={result.final_objective:.6f}  "
              f"iters={result.num_iterations}  time={result.total_time:.1f}s")
