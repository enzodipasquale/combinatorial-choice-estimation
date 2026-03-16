import numpy as np
import combest as ce
from combest.subproblems.registry.quadratic_obj.quadratic_supermodular.min_cut import (
    QuadraticSupermodularMinCutSolver,
)
from solver import TwoStageSolver
from oracles import build_oracles
from dc import DCSolver

BETA = 0.9
M = 15
R = 1
N_OBS = 1000
N_REV = 1
N_COV = N_REV + 3

THETA_TRUE = np.array([1.0] * N_REV + [-2.0, -1.0, .5])

SIGMA_1 = 1.0
SIGMA_2 = 1.0

SEED_DGP = 42
SEED_EST = 43

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
cov_oracle, err_oracle = build_oracles(dgp, beta=BETA, seed=SEED_DGP,
                                       sigma_1=SIGMA_1,
                                       sigma_2=SIGMA_2)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)

obs_b = dgp.subproblems.generate_obs_bundles(THETA_TRUE)
is_root = dgp.comm_manager.is_root()
if is_root:
    print(f"DGP: theta_true = {THETA_TRUE}")
    print(f"     beta={BETA}, M={M}, R={R}, N={N_OBS}")
    print(f"     obs items/firm: mean={obs_b.sum(1).mean():.2f}  "
          f"max={obs_b.sum(1).max()}")

names = [f"rev{i}" for i in range(N_REV)] + ["entry_c", "entry_dist", "syn"]

switch = 1 - state_chars
syn_state = state_chars @ syn_chars

modular = np.stack([
    rev_chars_1[:, 0, :],
    switch,
    switch * entry_chars[None, :],
    switch * syn_state * entry_chars[None, :],
], axis=-1)

eps_1_est = np.zeros((N_OBS, M))
for i in range(N_OBS):
    eps_1_est[i] = np.random.default_rng((SEED_EST, i, 0)).normal(0, SIGMA_1, M)


def build_mincut_model():
    m = ce.Model()
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
    cov_o, err_o = build_oracles(m, beta=BETA, seed=SEED_EST,
                                  sigma_1=SIGMA_1,
                                  sigma_2=SIGMA_2)
    m.subproblems.load_solver(TwoStageSolver)
    m.subproblems.initialize_solver()
    m.features.set_covariates_oracle(cov_o)
    m.features.set_error_oracle(err_o)
    return m


if is_root:
    print(f"\n{'='*60}")
    print(f"  1. MINCUT ROW GENERATION (QuadraticSupermodularMinCutSolver)")
    print(f"{'='*60}")

model_mc = build_mincut_model()
result_mc = model_mc.point_estimation.n_slack.solve(
    initialize_solver=False, verbose=True)

if is_root:
    print(f"\n  MINCUT theta_hat = {result_mc.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_mc.theta_hat[j]:+.4f}"
              f"  err={result_mc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_mc.final_objective:.6f}  "
          f"iters={result_mc.num_iterations}  time={result_mc.total_time:.1f}s")

if is_root:
    print(f"\n{'='*60}")
    print(f"  2. DC ALGORITHM (TwoStageSolver, beta={BETA})")
    print(f"{'='*60}")

model_dc = build_dc_model()
solver = model_dc.subproblems.subproblem_solver
row_gen = model_dc.point_estimation.n_slack
dc = DCSolver(row_gen, solver)

result_dc = dc.solve(THETA_TRUE * 0.5, max_dc_iters=30, tol=1e-6, verbose=True)

if is_root and result_dc is not None:
    print(f"\n  DC theta_hat = {result_dc.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_dc.theta_hat[j]:+.4f}"
              f"  err={result_dc.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_dc.final_objective:.6f}  "
          f"dc_iters={result_dc.num_iterations}  time={result_dc.total_time:.1f}s")

solver.solve_Q(THETA_TRUE)
f_true, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(THETA_TRUE)

theta_mc = result_mc.theta_hat if is_root else np.zeros(N_COV)
theta_mc = model_dc.comm_manager.Bcast(theta_mc)
solver.solve_Q(theta_mc)
f_mc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_mc)

theta_dc = result_dc.theta_hat if (is_root and result_dc is not None) else np.zeros(N_COV)
theta_dc = model_dc.comm_manager.Bcast(theta_dc)
solver.solve_Q(theta_dc)
f_dc, _ = model_dc.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_dc)

if is_root and result_dc is not None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON (all obj evaluated on 2SP model)")
    print(f"{'='*60}")
    print(f"  {'':>10}  {'true':>10}  {'mincut':>10}  {'DC':>10}")
    for j, name in enumerate(names):
        print(f"  {name:>10}  {THETA_TRUE[j]:+10.4f}  {result_mc.theta_hat[j]:+10.4f}  "
              f"{result_dc.theta_hat[j]:+10.4f}")
    print(f"  {'obj':>10}  {f_true:10.4f}  {f_mc:10.4f}  {f_dc:10.4f}")
