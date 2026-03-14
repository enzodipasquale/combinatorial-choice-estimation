"""R-scaling test with more seeds for reliable mean gradient estimates.

Focus on R = 10, 30, 100 with 30 seeds each.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

beta = 3
M, K = 5, 5
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
n_seeds = 30

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["g_rev", "g_s", "g_c"]
R_values = [10, 30, 100]

print(f"M={M}  K={K}  n_obs={n_obs}  beta={beta}  theta_true={theta_true}")
print(f"Averaging over {n_seeds} seeds. DGP seed={seed_dgp}")
print()

for R_val in R_values:
    g_list = []
    f_list = []
    for seed_idx in range(n_seeds):
        seed_est = 100 + seed_idx
        inp = {"id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
               "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                             "syn_chars": syn_chars, "beta": beta, "R": R_val, "seed": seed_dgp}}
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

        model = ce.Model()
        cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                                "n_covariates": n_cov, "n_simulations": 1},
                 "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
        model.load_config(cfg_e)
        model.data.load_and_distribute_input_data({
            "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                        "obs_bundles": obs_b},
            "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                          "syn_chars": syn_chars, "beta": beta, "R": R_val, "seed": seed_est}})
        co2, eo2 = build_oracles(model, seed=seed_est)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(co2)
        model.features.set_error_oracle(eo2)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
        f_list.append(f_val)
        g_list.append(g_val)

    g_arr = np.array(g_list)
    f_arr = np.array(f_list)
    mean_g = g_arr.mean(0)
    se_g = g_arr.std(0) / np.sqrt(n_seeds)  # standard error of mean

    print(f"R = {R_val}")
    print(f"  mean_f = {f_arr.mean():.4f} +/- {f_arr.std()/np.sqrt(n_seeds):.4f}")
    for j, n in enumerate(names):
        print(f"  mean_{n} = {mean_g[j]:+.6f} +/- {se_g[j]:.6f}  "
              f"(t = {abs(mean_g[j])/se_g[j]:.2f})")
    print(f"  |mean_g| = {np.linalg.norm(mean_g):.6f}")
    print()
