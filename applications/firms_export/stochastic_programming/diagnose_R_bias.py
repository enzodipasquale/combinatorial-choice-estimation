"""Test whether the gradient at theta_true shrinks with R.

Hypothesis: b_1_star from the joint MIP is "in-sample" for eps2.
The joint MIP picks b_1 that maximizes utility_1 + (beta/R) sum_r utility_2_r.
b_1_star is overfit to the specific R draws of eps2, biasing covariates.
If true, the gradient bias should decrease with R.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Setup: match user's M=15 but smaller n_obs for speed ──────────
beta = 3
M, K = 10, 10
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
seed_est = 43

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 1).astype(float)  # all zeros
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

R_values = [1, 5, 10, 30, 50, 100]
names = ["θ_rev", "θ_s", "θ_c"]

print(f"M={M}  K={K}  n_obs={n_obs}  beta={beta}  theta_true={theta_true}")
print(f"seed_dgp={seed_dgp}  seed_est={seed_est}")
print()
print(f"{'R':>6}  {'f':>10}  {'|g|':>10}  " + "  ".join(f"{n:>12}" for n in names)
      + "  flips  flip_rate")
print("-" * 90)

for R in R_values:
    # DGP with this R
    rng2 = np.random.default_rng(seed_dgp)  # reset rng for consistent chars
    input_data = {
        "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
        "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                      "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp},
    }
    cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
           "subproblem": {"gurobi_params": {"TimeLimit": 30}}}

    dgp = ce.Model()
    dgp.load_config(cfg)
    dgp.data.load_and_distribute_input_data(input_data)
    co, eo = build_oracles(dgp, seed=seed_dgp)
    dgp.subproblems.load_solver(TwoStageSolver)
    dgp.subproblems.initialize_solver()
    dgp.features.set_covariates_oracle(co)
    dgp.features.set_error_oracle(eo)
    obs_b = dgp.subproblems.generate_obs_bundles(theta_true)

    # Estimation with same R
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
    b1_diff = (pol["b_1_star"] != obs_b)
    n_flips = b1_diff.any(1).sum()
    flip_rate = b1_diff.mean()

    g_str = "  ".join(f"{g_val[j]:+12.4f}" for j in range(n_cov))
    print(f"{R:>6}  {f_val:10.4f}  {np.linalg.norm(g_val):10.4f}  {g_str}"
          f"  {n_flips:>5}  {flip_rate:.3f}")
