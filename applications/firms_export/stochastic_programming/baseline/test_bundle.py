import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Settings ──────────────────────────────────────────────────────────
beta = .8
M = 5
R_dgp = 50
R_est = 50
S_est = 1
n_obs = 500
n_rev = 1
n_cov = n_rev + 3
theta_true = np.array([.5] * n_rev + [-5.0, -1.0, 0.1])
sigma_eps = 1.0   # permanent component std
sigma_nu_1 = .5   # period-1 transitory component std
sigma_nu_2 = .5   # period-2 transitory component std
seed_dgp = 42
seed_est = 43
max_iters = 50
tau = 1


# ── Draw characteristics (shared across DGP and estimation) ────────
rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (n_obs, n_rev, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (n_obs, n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.9).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": beta, "R": R_dgp},
}
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M,
                   "n_covariates": n_cov},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

# ── Phase 1: DGP — generate observed bundles (n_simulations=1) ─────

dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
cov_oracle, err_oracle = build_oracles(dgp, seed=seed_dgp,
                                       sigma_eps=sigma_eps,
                                       sigma_nu_1=sigma_nu_1,
                                       sigma_nu_2=sigma_nu_2)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))

# ── Phase 2: Estimation (n_simulations=1, R=1) ────────────────────
model = ce.Model()
cfg["dimensions"]["n_simulations"] = S_est
input_data["id_data"]["obs_bundles"] = obs_b_dgp
input_data["item_data"]["R"] = R_est

model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)

cov_oracle, err_oracle = build_oracles(model, seed=seed_est,
                                       sigma_eps=sigma_eps,
                                       sigma_nu_1=sigma_nu_1,
                                       sigma_nu_2=sigma_nu_2)
solver = model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

# ── Run bundle solver ──────────────────────────────────────────────
is_root = model.comm_manager.is_root()
theta0 = theta_true.copy()
if is_root:
    print(f"theta_true = {theta_true}  R_dgp={R_dgp}  R_est={R_est}")

result = model.point_estimation.bundle.solve(
    theta0, tau=tau, max_iters=max_iters, verbose=True)

if is_root:
    err = np.linalg.norm(result.theta_hat - theta_true)
    print(f"\ntheta_hat  = {result.theta_hat}")
    print(f"err={err:.4f}  obj={result.final_objective:.6f}  "
          f"iters={result.num_iterations}  converged={result.converged}  "
          f"time={result.total_time:.1f}s")

# ── Evaluate objective at theta_hat and theta_true ────────────────
f_hat, g_hat = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(result.theta_hat)
f_true, g_true = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
if is_root:
    print(f"\nf(theta_hat)  = {f_hat:.6f}   |grad| = {np.linalg.norm(g_hat):.6f}")
    print(f"f(theta_true) = {f_true:.6f}   |grad| = {np.linalg.norm(g_true):.6f}")
