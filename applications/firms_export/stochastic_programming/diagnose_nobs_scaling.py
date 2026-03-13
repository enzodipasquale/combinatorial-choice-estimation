"""Test: does gradient at theta_true decrease with n_obs?

If |g| ~ 1/sqrt(N): it's variance (expected).
If |g| stays flat: there's a systematic bias that doesn't average out.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

beta = 3
M, K = 10, 10
R = 30
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
seed_est = 43

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["θ_rev", "θ_s", "θ_c"]
N_values = [50, 100, 200, 500, 1000, 2000]

print(f"M={M}  K={K}  R={R}  beta={beta}  theta_true={theta_true}")
print(f"seed_dgp={seed_dgp}  seed_est={seed_est}")
print()
hdr = (f"  {'N':>6}  {'f':>10}  " + "  ".join(f"{n:>10}" for n in names)
       + f"  {'|g|':>10}  {'|g|*√N':>10}  {'b1_diff':>8}")
print(hdr)
print("-" * len(hdr))

for n_obs in N_values:
    state_chars = np.zeros((n_obs, M))

    # DGP
    inp = {"id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
           "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                         "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp}}
    cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
           "subproblem": {"gurobi_params": {"TimeLimit": 30}}}
    dgp = ce.Model()
    dgp.load_config(cfg)
    dgp.data.load_and_distribute_input_data(inp)
    co, eo = build_oracles(dgp, seed=seed_dgp)
    dgp.subproblems.load_solver(TwoStageSolver)
    dgp.subproblems.initialize_solver()
    dgp.features.set_covariates_oracle(co)
    dgp.features.set_error_oracle(eo)
    obs_b = dgp.subproblems.generate_obs_bundles(theta_true)

    # Estimation
    model = ce.Model()
    cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                            "n_covariates": n_cov, "n_simulations": 1},
             "subproblem": {"gurobi_params": {"TimeLimit": 30}}}
    id_e = {"state_chars": state_chars, "capacity": np.full(n_obs, K),
            "obs_bundles": obs_b}
    item_e = {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
              "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_est}
    model.load_config(cfg_e)
    model.data.load_and_distribute_input_data({"id_data": id_e, "item_data": item_e})
    co2, eo2 = build_oracles(model, seed=seed_est)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(co2)
    model.features.set_error_oracle(eo2)

    f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    pol = model.data.local_data.id_data["policies"]
    n_diff = (pol["b_1_star"] != obs_b).any(1).sum()
    g_norm = np.linalg.norm(g_val)

    g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
    print(f"  {n_obs:>6}  {f_val:10.4f}  {g_str}  {g_norm:10.4f}"
          f"  {g_norm * np.sqrt(n_obs):10.4f}  {n_diff:>8}")

print()
print("If |g|*sqrt(N) is roughly constant → gradient is pure variance (CLT rate).")
print("If |g| is roughly constant → systematic bias that doesn't average out.")
