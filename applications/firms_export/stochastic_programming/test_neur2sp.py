"""End-to-end test: train the Neur2SP surrogate and compare with the exact solver.

Steps:
  1. Generate training data  (offline, Gurobi-only)
  2. Train the ReLU network  (offline, PyTorch)
  3. Run the bundle estimator with the original exact solver
  4. Run the bundle estimator with the NN surrogate solver
  5. Compare theta_hat, objectives, and timings

Usage (from stochastic_programming/):
    python test_neur2sp.py
"""
import time
import numpy as np
import combest as ce
from solver import TwoStageSolver
from neur2sp.solver import TwoStageSolverNN
from oracles import build_oracles

# ── Problem dimensions (same as test_bundle.py) ────────────────────
beta = 3
M, K = 3, 3
R_dgp = 100
R_est = 100
n_obs = 1000
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0] * n_rev + [-5.0, 0.5])
seed_dgp = 42
seed_est = 43
max_iters = 200
tau = 0.1

# ── Characteristics (shared across all phases) ─────────────────────
rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.9).astype(float)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

# ── Phase 0: Generate DGP observed bundles (exact solver) ──────────
print("=" * 60)
print("Phase 0: DGP — generate observed bundles")
print("=" * 60)
input_data_base = {
    "id_data": {"state_chars": state_chars,
                "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars,
                  "beta": beta, "R": R_dgp, "seed": seed_dgp},
}
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}
dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data_base)
cov_o, err_o = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_o)
dgp.features.set_error_oracle(err_o)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
print(f"  obs_b  sum(axis=1) range: "
      f"[{obs_b_dgp.sum(1).min()}, {obs_b_dgp.sum(1).max()}]")

# ── Phase 1: Offline — generate NN training data & train ───────────
print("\n" + "=" * 60)
print("Phase 1: Offline NN training pipeline")
print("=" * 60)

data_path = "neur2sp/data.npz"
model_path = "neur2sp/model.pt"

# 1a. generate data
from neur2sp.generate_data import generate_dataset
theta_bounds = {
    "theta_rev": (-5.0, 5.0),
    "theta_s": (-10.0, 0.0),
    "theta_c": (-1.0, 2.0),
}
print("Generating training data ...")
t0 = time.time()
inputs, labels = generate_dataset(
    rev_chars_2, syn_chars, beta, M, K, n_rev,
    theta_bounds, n_samples=3000, R_train=500, seed=123,
)
np.savez(data_path, inputs=inputs, labels=labels,
         rev_chars_2=rev_chars_2, syn_chars=syn_chars,
         M=M, K=K, n_rev=n_rev, beta=beta,
         theta_bounds_rev=theta_bounds["theta_rev"],
         theta_bounds_s=theta_bounds["theta_s"],
         theta_bounds_c=theta_bounds["theta_c"])
print(f"  Data generation: {time.time()-t0:.1f}s  ({len(labels)} samples)")

# 1b. train NN
from neur2sp.train import train as train_nn
print("Training NN ...")
t0 = time.time()
train_nn(data_path, model_path, hidden_dim=32, n_hidden=2,
         lr=1e-3, epochs=500, batch_size=256)
print(f"  Training: {time.time()-t0:.1f}s")

# ── Phase 2: Estimation with EXACT solver ──────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Bundle estimation — EXACT solver")
print("=" * 60)

input_data_exact = {
    "id_data": {"state_chars": state_chars,
                "capacity": np.full(n_obs, K),
                "obs_bundles": obs_b_dgp},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars,
                  "beta": beta, "R": R_est, "seed": seed_est},
}

model_exact = ce.Model()
model_exact.load_config(cfg)
model_exact.data.load_and_distribute_input_data(input_data_exact)
cov_o, err_o = build_oracles(model_exact, seed=seed_est)
model_exact.subproblems.load_solver(TwoStageSolver)
model_exact.subproblems.initialize_solver()
model_exact.features.set_covariates_oracle(cov_o)
model_exact.features.set_error_oracle(err_o)

t0 = time.time()
res_exact = model_exact.point_estimation.bundle.solve(
    theta_true.copy(), tau=tau, max_iters=max_iters, verbose=True)
time_exact = time.time() - t0

print(f"\n  theta_hat  = {res_exact.theta_hat}")
print(f"  error      = {np.linalg.norm(res_exact.theta_hat - theta_true):.4f}")
print(f"  objective  = {res_exact.final_objective:.6f}")
print(f"  iters      = {res_exact.num_iterations}")
print(f"  time       = {time_exact:.1f}s")

# ── Phase 3: Estimation with NN SURROGATE solver ───────────────────
print("\n" + "=" * 60)
print("Phase 3: Bundle estimation — NN SURROGATE solver")
print("=" * 60)

input_data_nn = {
    "id_data": {"state_chars": state_chars,
                "capacity": np.full(n_obs, K),
                "obs_bundles": obs_b_dgp},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars,
                  "beta": beta, "R": R_est, "seed": seed_est,
                  "nn_model_path": model_path},
}

model_nn = ce.Model()
model_nn.load_config(cfg)
model_nn.data.load_and_distribute_input_data(input_data_nn)
cov_o, err_o = build_oracles(model_nn, seed=seed_est)
model_nn.subproblems.load_solver(TwoStageSolverNN)
model_nn.subproblems.initialize_solver()
model_nn.features.set_covariates_oracle(cov_o)
model_nn.features.set_error_oracle(err_o)

t0 = time.time()
res_nn = model_nn.point_estimation.bundle.solve(
    theta_true.copy(), tau=tau, max_iters=max_iters, verbose=True)
time_nn = time.time() - t0

print(f"\n  theta_hat  = {res_nn.theta_hat}")
print(f"  error      = {np.linalg.norm(res_nn.theta_hat - theta_true):.4f}")
print(f"  objective  = {res_nn.final_objective:.6f}")
print(f"  iters      = {res_nn.num_iterations}")
print(f"  time       = {time_nn:.1f}s")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
err_e = np.linalg.norm(res_exact.theta_hat - theta_true)
err_n = np.linalg.norm(res_nn.theta_hat - theta_true)
print(f"  {'':20s} {'EXACT':>12s} {'NN-SURR':>12s}")
print(f"  {'theta error':20s} {err_e:12.4f} {err_n:12.4f}")
print(f"  {'objective':20s} {res_exact.final_objective:12.6f} {res_nn.final_objective:12.6f}")
print(f"  {'iterations':20s} {res_exact.num_iterations:12d} {res_nn.num_iterations:12d}")
print(f"  {'time (s)':20s} {time_exact:12.1f} {time_nn:12.1f}")
