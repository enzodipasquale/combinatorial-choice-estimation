import sys
import yaml
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

with open(Path(__file__).resolve().parent / "test_config.yaml") as f:
    CFG = yaml.safe_load(f)["test_dc"]

M = CFG["M"]
R = CFG["R"]
N_OBS = CFG["n_obs"]
N_REV = CFG["n_rev"]
N_COV_STATIC = N_REV + 4
N_COV = 2 * N_COV_STATIC
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

input_data_dgp = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "R": R},
}
cfg_dgp = {
    "dimensions": {"n_obs": N_OBS, "n_items": M, "n_covariates": N_COV},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

dgp = ce.Model()
dgp.load_config(cfg_dgp)
dgp.data.load_and_distribute_input_data(input_data_dgp)
cov_oracle, err_oracle = build_oracles(dgp, seed=SEED_DGP,
                                       sigma_1=SIGMA_1,
                                       sigma_2=SIGMA_2)
dgp.subproblems.load_solver(TwoStageSolverSplit)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)

obs_b = dgp.subproblems.generate_obs_bundles(THETA_TRUE)
is_root = dgp.comm_manager.is_root()
if is_root:
    print(f"DGP: theta_true = {THETA_TRUE}")
    print(f"     M={M}, R={R}, N={N_OBS}")
    print(f"     obs items/firm: mean={obs_b.sum(1).mean():.2f}  "
          f"max={obs_b.sum(1).max()}")

names_1 = [f"rev{i}" for i in range(N_REV)] + ["entry_c", "entry_dist", "syn", "syn_d"]
names = [n + "_1" for n in names_1] + [n + "_2" for n in names_1]

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


def build_mincut_model():
    m = ce.Model()
    cfg_mc = {
        "dimensions": {"n_obs": N_OBS, "n_items": M,
                       "n_covariates": N_COV_STATIC, "n_simulations": 1},
        "row_generation": {"max_iters": 200, "tolerance": 1e-6,
                           "theta_bounds": {"lb": -1000}},
    }
    input_data_mc = {
        "id_data": {"obs_bundles": obs_b, "modular": modular,
                    "constraint_mask": None},
        "item_data": {},
    }
    m.load_config(cfg_mc)
    m.data.load_and_distribute_input_data(input_data_mc)

    local_ids = m.comm_manager.agent_ids
    local_eps = eps_1_est[local_ids]
    m.features.local_modular_errors = local_eps
    m.features._error_oracle = lambda bundles, ids: (local_eps[ids] * bundles).sum(-1)
    m.features._error_oracle_takes_data = False
    m.data_manager.id_data = m.data_manager.local_data.id_data
    m.features.build_quadratic_covariates_from_data()

    m.subproblems.load_solver(QuadraticSupermodularMinCutSolver)
    m.subproblems.initialize_solver()
    return m


def build_dc_model():
    m = ce.Model()
    cfg_est = {
        "dimensions": {"n_obs": N_OBS, "n_items": M,
                       "n_covariates": N_COV, "n_simulations": 1},
        "subproblem": {"gurobi_params": {"TimeLimit": 10}},
        "row_generation": {"max_iters": 200, "tolerance": 1e-6,
                           "theta_bounds": {"lb": -1000}},
    }
    input_data_est = {
        "id_data": {"state_chars": state_chars,
                    "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                    "obs_bundles": obs_b},
        "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                      "R": R},
    }
    m.load_config(cfg_est)
    m.data.load_and_distribute_input_data(input_data_est)
    cov_o, err_o = build_oracles(m, seed=SEED_EST,
                                  sigma_1=SIGMA_1,
                                  sigma_2=SIGMA_2)
    m.subproblems.load_solver(TwoStageSolverSplit)
    m.subproblems.initialize_solver()
    m.features.set_covariates_oracle(cov_o)
    m.features.set_error_oracle(err_o)
    return m


if is_root:
    print(f"\n{'='*60}")
    print(f"  1. MINCUT ROW GENERATION (static, {N_COV_STATIC} params)")
    print(f"{'='*60}")

model_mc = build_mincut_model()
result_mc = model_mc.point_estimation.n_slack.solve(
    initialize_solver=False, verbose=True)

if is_root:
    print(f"\n  MINCUT theta_hat = {result_mc.theta_hat}")
    for j, name in enumerate(names_1):
        print(f"    {name:>10}:  true_1={THETA_TRUE[j]:+.4f}  hat={result_mc.theta_hat[j]:+.4f}"
              f"  err={result_mc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_mc.final_objective:.6f}  "
          f"iters={result_mc.num_iterations}  time={result_mc.total_time:.1f}s")

if is_root:
    print(f"\n{'='*60}")
    print(f"  2. DC ALGORITHM (TwoStageSolverSplit, {N_COV} params)")
    print(f"{'='*60}")

theta0_dc = THETA_TRUE * 0.5
model_dc = build_dc_model()
solver = model_dc.subproblems.subproblem_solver
row_gen = model_dc.point_estimation.n_slack
dc = DCSolver(row_gen, solver)

result_dc = dc.solve(theta0_dc, max_dc_iters=30, tol=1e-6, verbose=True)

if is_root and result_dc is not None:
    print(f"\n  DC theta_hat = {result_dc.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_dc.theta_hat[j]:+.4f}"
              f"  err={result_dc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_dc.final_objective:.6f}  "
          f"dc_iters={result_dc.num_iterations}  time={result_dc.total_time:.1f}s")

solver.solve_Q(THETA_TRUE)
f_true, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(THETA_TRUE)

theta_mc = np.zeros(N_COV)
if is_root:
    theta_mc[:N_COV_STATIC] = result_mc.theta_hat
    theta_mc[N_COV_STATIC:] = result_mc.theta_hat
theta_mc = model_dc.comm_manager.Bcast(theta_mc)
solver.solve_Q(theta_mc)
f_mc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_mc)

theta_dc = result_dc.theta_hat if (is_root and result_dc is not None) else np.zeros(N_COV)
theta_dc = model_dc.comm_manager.Bcast(theta_dc)
solver.solve_Q(theta_dc)
f_dc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_dc)

if is_root and result_dc is not None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON (all obj evaluated on split 2SP model)")
    print(f"{'='*60}")
    print(f"  {'':>10}  {'true':>10}  {'mincut':>10}  {'DC':>10}")
    for j, name in enumerate(names):
        mc_val = theta_mc[j]
        print(f"  {name:>10}  {THETA_TRUE[j]:+10.4f}  {mc_val:+10.4f}  "
              f"{result_dc.theta_hat[j]:+10.4f}")
    print(f"  {'obj':>10}  {f_true:10.4f}  {f_mc:10.4f}  {f_dc:10.4f}")
