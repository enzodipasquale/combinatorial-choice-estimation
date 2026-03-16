import sys
from pathlib import Path
import numpy as np
import combest as ce

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solver import TwoStageSolver
from oracles import build_oracles

BETA = 0.8
M = 5
R_DGP = 50
R_EST = 50
N_OBS = 500
N_REV = 1
N_COV = N_REV + 3
S_EST = 1

THETA_TRUE = np.array([0.5] * N_REV + [-5.0, -1.0, 0.1])

SIGMA_1 = 1.0
SIGMA_2 = 0.5

SEED_DGP = 42
SEED_EST = 43
MAX_ITERS = 50
TAU = 1

rng = np.random.default_rng(SEED_DGP)
rev_base = rng.uniform(0, 1.0, (N_REV, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
state_chars = (rng.random((N_OBS, M)) > 0.9).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": BETA, "R": R_DGP},
}
cfg = {
    "dimensions": {"n_obs": N_OBS, "n_items": M,
                   "n_covariates": N_COV},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
cov_oracle, err_oracle = build_oracles(dgp, seed=SEED_DGP,
                                       sigma_1=SIGMA_1,
                                       sigma_2=SIGMA_2)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(THETA_TRUE)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))

model = ce.Model()
cfg["dimensions"]["n_simulations"] = S_EST
input_data["id_data"]["obs_bundles"] = obs_b_dgp
input_data["item_data"]["R"] = R_EST

model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)

cov_oracle, err_oracle = build_oracles(model, seed=SEED_EST,
                                       sigma_1=SIGMA_1,
                                       sigma_2=SIGMA_2)
solver = model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

is_root = model.comm_manager.is_root()
theta0 = THETA_TRUE.copy()
if is_root:
    print(f"theta_true = {THETA_TRUE}  R_dgp={R_DGP}  R_est={R_EST}")

result = model.point_estimation.bundle.solve(
    theta0, tau=TAU, max_iters=MAX_ITERS, verbose=True)

if is_root:
    err = np.linalg.norm(result.theta_hat - THETA_TRUE)
    print(f"\ntheta_hat  = {result.theta_hat}")
    print(f"err={err:.4f}  obj={result.final_objective:.6f}  "
          f"iters={result.num_iterations}  converged={result.converged}  "
          f"time={result.total_time:.1f}s")

f_hat, g_hat = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(result.theta_hat)
f_true, g_true = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(THETA_TRUE)
if is_root:
    print(f"\nf(theta_hat)  = {f_hat:.6f}   |grad| = {np.linalg.norm(g_hat):.6f}")
    print(f"f(theta_true) = {f_true:.6f}   |grad| = {np.linalg.norm(g_true):.6f}")
