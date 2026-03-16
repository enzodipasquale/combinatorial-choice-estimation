import sys
import time
from pathlib import Path
import numpy as np
import combest as ce

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
from solver import TwoStageSolver
from oracles import build_oracles
from neur2sp.solver import TwoStageSolverNN

beta = 0.8
M = 3
R_dgp, R_est = 20, 20
n_obs, n_rev = 100, 1
n_cov = n_rev + 3
theta_true = np.array([0.5] * n_rev + [-5.0, -1.0, 0.1])
sigma_eps, sigma_nu_1, sigma_nu_2 = 1.0, 0.5, 0.5
seed_dgp, seed_est = 42, 43
max_iters, tau = 50, 1.0

rng = np.random.default_rng(seed_dgp)
rev_chars_1 = rng.uniform(0, 2, (n_obs, n_rev, M))
rev_chars_2 = rng.uniform(0, 2, (n_obs, n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.9).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

input_data_base = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": beta, "R": R_dgp},
}
dgp = ce.Model()
is_root = dgp.comm_manager.is_root()

if is_root:
    print("=" * 60)
    print("Phase 0: DGP")
    print("=" * 60)

dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data_base)
cov_o, err_o = build_oracles(dgp, seed=seed_dgp,
                              sigma_eps=sigma_eps,
                              sigma_nu_1=sigma_nu_1,
                              sigma_nu_2=sigma_nu_2)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_o)
dgp.features.set_error_oracle(err_o)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
if is_root:
    print(f"  obs_b sum range: [{obs_b_dgp.sum(1).min()}, {obs_b_dgp.sum(1).max()}]")

if is_root:
    print("\n" + "=" * 60)
    print("Phase 1: Offline NN training")
    print("=" * 60)

data_path, model_path = "neur2sp/data.npz", "neur2sp/model.pt"
perpetual = 1 / (1 - beta)
beta_perpetual = beta * perpetual

theta_rev_range = (-5.0, 5.0)
max_rev = rev_chars_2.max()
eff_rev_ub = beta_perpetual * (abs(theta_rev_range[1]) * max_rev + 3 * sigma_eps)
eff_rev_lb = -eff_rev_ub
theta_bounds = {"theta_s": (-10.0, 0.0), "theta_sc": (-5.0, 0.0),
                "theta_c": (-1.0, 2.0)}
eff_rev_bounds = (eff_rev_lb, eff_rev_ub)

from neur2sp.generate_data import generate_dataset
if is_root:
    print(f"Generating training data (eff_rev_bounds=[{eff_rev_lb:.1f}, {eff_rev_ub:.1f}]) ...")
t0 = time.time()
inputs, labels = generate_dataset(
    entry_chars, syn_chars, beta, beta_perpetual, sigma_nu_2,
    M, theta_bounds, eff_rev_bounds,
    n_samples=1000, R_train=200, seed=123)
np.savez(data_path, inputs=inputs, labels=labels,
         entry_chars=entry_chars, syn_chars=syn_chars,
         M=M, beta=beta, beta_perpetual=beta_perpetual,
         sigma_nu_2=sigma_nu_2,
         eff_rev_bounds=np.array(eff_rev_bounds),
         theta_bounds_s=theta_bounds["theta_s"],
         theta_bounds_sc=theta_bounds["theta_sc"],
         theta_bounds_c=theta_bounds["theta_c"])
if is_root:
    print(f"  Data: {time.time()-t0:.1f}s  ({len(labels)} samples)")

from neur2sp.train import train as train_nn
if is_root:
    print("Training NN ...")
t0 = time.time()
train_nn(data_path, model_path, hidden_dim=32, n_hidden=2,
         lr=1e-3, epochs=500, batch_size=256)
if is_root:
    print(f"  Training: {time.time()-t0:.1f}s")

if is_root:
    print("\n" + "=" * 60)
    print("Phase 2: EXACT solver")
    print("=" * 60)
input_data_est = {
    "id_data": {"state_chars": state_chars,
                "obs_bundles": obs_b_dgp,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": beta, "R": R_est},
}

model_exact = ce.Model()
model_exact.load_config(cfg)
model_exact.data.load_and_distribute_input_data(input_data_est)
cov_o, err_o = build_oracles(model_exact, seed=seed_est,
                              sigma_eps=sigma_eps,
                              sigma_nu_1=sigma_nu_1,
                              sigma_nu_2=sigma_nu_2)
model_exact.subproblems.load_solver(TwoStageSolver)
model_exact.subproblems.initialize_solver()
model_exact.features.set_covariates_oracle(cov_o)
model_exact.features.set_error_oracle(err_o)

t0 = time.time()
res_exact = model_exact.point_estimation.bundle.solve(
    theta_true.copy(), tau=tau, max_iters=max_iters, verbose=True)
time_exact = time.time() - t0

if is_root:
    print("\n" + "=" * 60)
    print("Phase 3: NN SURROGATE solver")
    print("=" * 60)
input_data_nn = {
    "id_data": {"state_chars": state_chars,
                "obs_bundles": obs_b_dgp,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": beta, "R": R_est,
                  "nn_model_path": model_path},
}

model_nn = ce.Model()
model_nn.load_config(cfg)
model_nn.data.load_and_distribute_input_data(input_data_nn)
cov_o, err_o = build_oracles(model_nn, seed=seed_est,
                              sigma_eps=sigma_eps,
                              sigma_nu_1=sigma_nu_1,
                              sigma_nu_2=sigma_nu_2)
model_nn.subproblems.load_solver(TwoStageSolverNN)
model_nn.subproblems.initialize_solver()
model_nn.features.set_covariates_oracle(cov_o)
model_nn.features.set_error_oracle(err_o)

t0 = time.time()
res_nn = model_nn.point_estimation.bundle.solve(
    theta_true.copy(), tau=tau, max_iters=max_iters, verbose=True)
time_nn = time.time() - t0

if is_root:
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    err_e = np.linalg.norm(res_exact.theta_hat - theta_true)
    err_n = np.linalg.norm(res_nn.theta_hat - theta_true)
    print(f"  {'':20s} {'EXACT':>12s} {'NN-SURR':>12s}")
    print(f"  {'theta error':20s} {err_e:12.4f} {err_n:12.4f}")
    print(f"  {'objective':20s} {res_exact.final_objective:12.6f} "
          f"{res_nn.final_objective:12.6f}")
    print(f"  {'iterations':20s} {res_exact.num_iterations:12d} "
          f"{res_nn.num_iterations:12d}")
    print(f"  {'time (s)':20s} {time_exact:12.1f} {time_nn:12.1f}")
