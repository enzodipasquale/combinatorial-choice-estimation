import gc
import yaml
from pathlib import Path
import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from solver import TwoStageSolverCF
from oracles import build_oracles
from dc import DCSolver
from combest.estimation.callbacks import adaptive_gurobi_timeout

with open(Path(__file__).resolve().parent / "config_test.yaml") as f:
    CFG = yaml.safe_load(f)["test_dc"]

BETA = CFG["beta"]
M = CFG["M"]
N_OBS = CFG["n_obs"]
N_REV = CFG["n_rev"]
N_COV = N_REV + 4
THETA_TRUE = np.array(CFG["theta_true"])
SIGMA_1 = CFG["sigma_1"]
SIGMA_2 = CFG["sigma_2"]
SEED_DGP = CFG["seed_dgp"]
SEED_EST = CFG["seed_est"]

rng = np.random.default_rng(SEED_DGP)
rev_base = rng.uniform(0, 1.0, (N_REV, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
state_chars = (rng.random((N_OBS, M)) > 0.5).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = [f"rev{i}" for i in range(N_REV)] + ["entry_c", "entry_dist", "syn", "syn_d"]

# -- DGP (closed-form solver) ------------------------------------------------
input_data_dgp = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars},
}
cfg_dgp = {
    "dimensions": {"n_obs": N_OBS, "n_items": M, "n_covariates": N_COV},
    "subproblem": {"gurobi_params": {"TimeLimit": 30}},
}

dgp = ce.Model()
dgp.load_config(cfg_dgp)
dgp.data.load_and_distribute_input_data(input_data_dgp)
cov_oracle, err_oracle = build_oracles(dgp, beta=BETA, seed=SEED_DGP,
                                       sigma_1=SIGMA_1, sigma_2=SIGMA_2)
dgp.subproblems.load_solver(TwoStageSolverCF)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)

obs_b = dgp.subproblems.generate_obs_bundles(THETA_TRUE)
is_root = dgp.comm_manager.is_root()
if is_root:
    print(f"DGP: theta_true = {THETA_TRUE}")
    print(f"     beta={BETA}, M={M}, N={N_OBS}, sigma_2={SIGMA_2}")
    print(f"     obs items/firm: mean={obs_b.sum(1).mean():.2f}  "
          f"max={obs_b.sum(1).max()}")

del dgp, cov_oracle, err_oracle
gc.collect()

# -- Static covariates -------------------------------------------------------
switch = 1 - state_chars
syn_state = state_chars @ syn_chars

modular = np.stack([
    rev_chars_1[:, 0, :],
    switch,
    switch * entry_chars[None, :],
    switch * syn_state,
    switch * syn_state * entry_chars[None, :],
], axis=-1)

eps_1_est = np.zeros((N_OBS, M))
for i in range(N_OBS):
    eps_1_est[i] = np.random.default_rng((SEED_EST, i, 0)).normal(0, SIGMA_1, M)

# -- MinCut -------------------------------------------------------------------
if is_root:
    print(f"\n{'='*60}")
    print(f"  1. MINCUT ROW GENERATION (QuadraticSupermodularMinCutSolver)")
    print(f"{'='*60}")

model_mc = ce.Model()
cfg_mc = {
    "dimensions": {"n_obs": N_OBS, "n_items": M,
                   "n_covariates": N_COV, "n_simulations": 1},
    "row_generation": {"max_iters": 200, "tolerance": 1e-6,
                       "theta_bounds": {"lb": -1000}},
}
input_data_mc = {
    "id_data": {"obs_bundles": obs_b, "modular": modular,
                "constraint_mask": None},
    "item_data": {},
}
model_mc.load_config(cfg_mc)
model_mc.data.load_and_distribute_input_data(input_data_mc)

local_ids = model_mc.comm_manager.agent_ids
local_eps = eps_1_est[local_ids]
model_mc.features.local_modular_errors = local_eps
model_mc.features._error_oracle = lambda bundles, ids: (local_eps[ids] * bundles).sum(-1)
model_mc.features._error_oracle_takes_data = False
model_mc.data_manager.id_data = model_mc.data_manager.local_data.id_data
model_mc.features.build_quadratic_covariates_from_data()

model_mc.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
model_mc.subproblems.initialize_solver()

result_mc = model_mc.point_estimation.n_slack.solve(
    initialize_solver=False, verbose=True)

if is_root:
    print(f"\n  MINCUT theta_hat = {result_mc.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_mc.theta_hat[j]:+.4f}"
              f"  err={result_mc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_mc.final_objective:.6f}  "
          f"iters={result_mc.num_iterations}  time={result_mc.total_time:.1f}s")

theta_mc_hat = result_mc.theta_hat.copy() if is_root else np.zeros(N_COV)

del model_mc, result_mc
gc.collect()

# -- DC (closed-form solver) --------------------------------------------------
if is_root:
    print(f"\n{'='*60}")
    print(f"  2. DC ALGORITHM (TwoStageSolverCF, beta={BETA})")
    print(f"{'='*60}")

model_dc = ce.Model()
cfg_est = {
    "dimensions": {"n_obs": N_OBS, "n_items": M,
                   "n_covariates": N_COV, "n_simulations": 1},
    "subproblem": {"gurobi_params": {"TimeLimit": 30}},
    "row_generation": {"max_iters": 200, "tolerance": 1e-6,
                       "theta_bounds": {"lb": -1000}},
}
input_data_est = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                "obs_bundles": obs_b},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars},
}
model_dc.load_config(cfg_est)
model_dc.data.load_and_distribute_input_data(input_data_est)
cov_o, err_o = build_oracles(model_dc, beta=BETA, seed=SEED_EST,
                              sigma_1=SIGMA_1, sigma_2=SIGMA_2)
model_dc.subproblems.load_solver(TwoStageSolverCF)
model_dc.subproblems.initialize_solver()
model_dc.features.set_covariates_oracle(cov_o)
model_dc.features.set_error_oracle(err_o)

solver = model_dc.subproblems.subproblem_solver
row_gen = model_dc.point_estimation.n_slack
dc = DCSolver(row_gen, solver)

rg_cb, _ = adaptive_gurobi_timeout(CFG["callbacks"]["row_gen"])
result_dc = dc.solve(THETA_TRUE * 0.5, max_dc_iters=30, tol=1e-6,
                     iteration_callback=rg_cb, verbose=True)

if is_root and result_dc is not None:
    print(f"\n  DC theta_hat = {result_dc.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_dc.theta_hat[j]:+.4f}"
              f"  err={result_dc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_dc.final_objective:.6f}  "
          f"dc_iters={result_dc.num_iterations}  time={result_dc.total_time:.1f}s")

# -- Comparison ---------------------------------------------------------------
solver.solve_Q(THETA_TRUE)
f_true, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(THETA_TRUE)

theta_mc_hat = model_dc.comm_manager.Bcast(theta_mc_hat)
solver.solve_Q(theta_mc_hat)
f_mc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_mc_hat)

theta_dc = result_dc.theta_hat if (is_root and result_dc is not None) else np.zeros(N_COV)
theta_dc = model_dc.comm_manager.Bcast(theta_dc)
solver.solve_Q(theta_dc)
f_dc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_dc)

if is_root and result_dc is not None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON (all obj evaluated on CF model)")
    print(f"{'='*60}")
    print(f"  {'':>10}  {'true':>10}  {'mincut':>10}  {'DC':>10}")
    for j, name in enumerate(names):
        print(f"  {name:>10}  {THETA_TRUE[j]:+10.4f}  {theta_mc_hat[j]:+10.4f}  "
              f"{result_dc.theta_hat[j]:+10.4f}")
    print(f"  {'obj':>10}  {f_true:10.4f}  {f_mc:10.4f}  {f_dc:10.4f}")
