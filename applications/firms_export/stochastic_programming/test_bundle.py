"""Estimate a 4-parameter two-stage stochastic programming model
using the Schramm-Zowe proximal bundle method.

    theta = [theta_rev1, theta_rev2, theta_entry, theta_synergy]

DGP generates observed bundles with n_simulations=1 (one error draw
per observation).  Estimation uses n_simulations=2 for Monte-Carlo
smoothing of the DC objective.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Problem dimensions ──────────────────────────────────────────────
beta = 4
M, R, K = 5, 20, 3
n_obs = 80
n_rev = 2
n_cov = n_rev + 2
theta_true = np.array([1.0, -0.5, -0.5, 0.2])

# ── Draw characteristics (shared across DGP and estimation) ────────
seed_dgp = 42
rng = np.random.default_rng(seed_dgp)
rev_chars = rng.uniform(0.5, 2.0, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.5).astype(float)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

# ── Phase 1: DGP — generate observed bundles (n_simulations=1) ─────
dgp = ce.Model()
dgp.load_config({
    "dimensions": {"n_obs": n_obs, "n_items": M,
                   "n_covariates": n_cov, "n_simulations": 1},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
})
dgp.data.load_and_distribute_input_data({
    "id_data": {"obs_bundles": np.zeros((n_obs, M), dtype=bool),
                "state_chars": state_chars, "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars": rev_chars, "syn_chars": syn_chars,
                  "beta": beta, "R": R, "seed": seed_dgp},
})
cov_oracle, err_oracle = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)

obs_b_dgp = dgp.subproblems.solve(theta_true).copy()  # (n_obs, M)

# ── Phase 2: Estimation (n_simulations=2) ──────────────────────────
model = ce.Model()
model.load_config({
    "dimensions": {"n_obs": n_obs, "n_items": M,
                   "n_covariates": n_cov, "n_simulations": 2},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
})
model.data.load_and_distribute_input_data({
    "id_data": {"obs_bundles": np.zeros((n_obs, M), dtype=bool),
                "state_chars": state_chars, "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars": rev_chars, "syn_chars": syn_chars,
                  "beta": beta, "R": R, "seed": seed_dgp},
})

seed_est = 123
cov_oracle, err_oracle = build_oracles(model, seed=seed_est)
solver = model.subproblems.load_solver(TwoStageSolver)
model.subproblems.initialize_solver()
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

# Replicate DGP bundles across simulations via obs_ids
obs_ids = model.comm_manager.obs_ids
obs_b = obs_b_dgp[obs_ids]
solver.obs_b = obs_b.astype(float)
model.data.local_data.id_data["obs_bundles"] = obs_b

# ── Run bundle solver ──────────────────────────────────────────────
is_root = model.comm_manager.is_root()
theta0 = np.zeros(n_cov)
if is_root:
    print(f"theta_true = {theta_true}")

result = model.point_estimation.bundle.solve(
    theta0, tau=1.0, max_iters=200, verbose=True)

if is_root:
    err = np.linalg.norm(result.theta_hat - theta_true)
    print(f"\ntheta_hat  = {result.theta_hat}")
    print(f"err={err:.4f}  obj={result.final_objective:.6f}  "
          f"iters={result.num_iterations}  converged={result.converged}  "
          f"time={result.total_time:.1f}s")
