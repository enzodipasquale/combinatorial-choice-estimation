"""Confirm: theta_c gradient is zero at beta=1 because theta_s=-5 is too large.

With smaller |theta_s|, entry is cheaper → larger bundles → synergies kick in.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

M, K = 10, 8
R = 30
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
seed_dgp = 42
seed_est = 43
beta = 1.0

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["θ_rev", "θ_s", "θ_c"]
print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}")
print()
hdr = (f"  {'θ_s':>8}  {'f':>10}  " + "  ".join(f"{n:>10}" for n in names)
       + f"  {'|g|':>10}  {'b1_diff':>8}  {'avg_sz':>8}")
print(hdr)
print("-" * len(hdr))

for theta_s_val in [-5.0, -2.0, -1.0, -0.5, -0.1, 0.0]:
    theta_true = np.array([1.0, theta_s_val, 0.1])

    inp = {"id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
           "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                         "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp}}
    cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
           "subproblem": {"gurobi_params": {"TimeLimit": 10}}}

    dgp = ce.Model()
    dgp.load_config(cfg)
    dgp.data.load_and_distribute_input_data(inp)
    co, eo = build_oracles(dgp, seed=seed_dgp)
    dgp.subproblems.load_solver(TwoStageSolver)
    dgp.subproblems.initialize_solver()
    dgp.features.set_covariates_oracle(co)
    dgp.features.set_error_oracle(eo)
    obs_b = dgp.subproblems.generate_obs_bundles(theta_true)
    avg_sz = obs_b.sum(1).mean()

    model = ce.Model()
    cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                            "n_covariates": n_cov, "n_simulations": 1},
             "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
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

    g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
    print(f"  {theta_s_val:8.1f}  {f_val:10.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}"
          f"  {n_diff:>8}  {avg_sz:8.1f}")
