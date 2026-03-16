import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles
from dc import DCSolver

BETA = 0.0
M = 10
R = 1
N_OBS = 1000
N_REV = 1
N_COV = N_REV + 3

THETA_TRUE = np.array([1.0] * N_REV + [-5.0, -1.0, 0.1])

SIGMA_EPS = 1.0
SIGMA_NU_1 = 1.0
SIGMA_NU_2 = 1.0

SEED_DGP = 42
SEED_EST = 99

rng = np.random.default_rng(SEED_DGP)
rev_base = rng.uniform(0, 1.0, (N_REV, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
state_chars = (rng.random((N_OBS, M)) > 0.1).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data_dgp = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": BETA, "R": R},
}
cfg_dgp = {
    "dimensions": {"n_obs": N_OBS, "n_items": M, "n_covariates": N_COV},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

dgp = ce.Model()
dgp.load_config(cfg_dgp)
dgp.data.load_and_distribute_input_data(input_data_dgp)
cov_oracle, err_oracle = build_oracles(dgp, seed=SEED_DGP,
                                       sigma_eps=SIGMA_EPS,
                                       sigma_nu_1=SIGMA_NU_1,
                                       sigma_nu_2=SIGMA_NU_2)
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


def build_est_model():
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
                      "beta": BETA, "R": R},
    }
    m.load_config(cfg_est)
    m.data.load_and_distribute_input_data(input_data_est)
    cov_o, err_o = build_oracles(m, seed=SEED_EST,
                                  sigma_eps=SIGMA_EPS,
                                  sigma_nu_1=SIGMA_NU_1,
                                  sigma_nu_2=SIGMA_NU_2)
    m.subproblems.load_solver(TwoStageSolver)
    m.subproblems.initialize_solver()
    m.features.set_covariates_oracle(cov_o)
    m.features.set_error_oracle(err_o)
    return m


if is_root:
    print(f"\n{'='*60}")
    print(f"  STATIC ROW GENERATION (baseline)")
    print(f"{'='*60}")

model_static = build_est_model()
result_static = model_static.point_estimation.n_slack.solve(
    initialize_solver=False, verbose=True)

if is_root:
    print(f"\n  STATIC theta_hat = {result_static.theta_hat}")
    for j, name in enumerate(names):
        print(f"    {name:>10}:  true={THETA_TRUE[j]:+.4f}  hat={result_static.theta_hat[j]:+.4f}"
              f"  err={result_static.theta_hat[j]-THETA_TRUE[j]:+.4f}")
    print(f"  obj={result_static.final_objective:.6f}  "
          f"iters={result_static.num_iterations}  time={result_static.total_time:.1f}s")

if is_root:
    print(f"\n{'='*60}")
    print(f"  DC ALGORITHM")
    print(f"{'='*60}")

model_dc = build_est_model()
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

if is_root and result_dc is not None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'':>10}  {'true':>10}  {'static':>10}  {'DC':>10}")
    for j, name in enumerate(names):
        print(f"  {name:>10}  {THETA_TRUE[j]:+10.4f}  {result_static.theta_hat[j]:+10.4f}  "
              f"{result_dc.theta_hat[j]:+10.4f}")
    print(f"  {'obj':>10}  {'':>10}  {result_static.final_objective:10.4f}  "
          f"{result_dc.final_objective:10.4f}")
