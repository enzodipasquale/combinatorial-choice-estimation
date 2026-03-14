"""Definitive test: does the gradient bias at theta_true decrease with R?

Hypothesis: b_1_star from the joint MIP is correlated with the eps_2 draws
used for the SAA continuation value. This correlation biases the period-2
covariates in the V term (but not Q, since hat_b_1 is independent of eps_2_est).

Prediction:
- |grad| at theta_true decreases with R  (SAA bias vanishes as R -> inf)
- |grad| does NOT decrease with n_obs     (bias is per-agent, not sampling noise)
- The bias is proportional to beta        (only period-2 covariates are affected)

We average over multiple estimation seeds to isolate the MEAN gradient (bias)
from the variance.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Setup ──────────────────────────────────────────────────────────────
beta = 3
M, K = 5, 5
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
n_seeds = 10  # number of estimation seeds to average over

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["g_rev", "g_s", "g_c"]
R_values = [5, 10, 30, 50, 100]

print(f"M={M}  K={K}  n_obs={n_obs}  beta={beta}  theta_true={theta_true}")
print(f"Averaging gradient over {n_seeds} estimation seeds per R value")
print(f"DGP seed={seed_dgp} (DGP also uses same R as estimation)")
print()

hdr = (f"  {'R':>5}  {'mean_f':>10}  " + "  ".join(f"{'mean_'+n:>12}" for n in names)
       + f"  {'|mean_g|':>12}  {'std_|g|':>10}")
print(hdr)
print("-" * len(hdr))

for R_val in R_values:
    f_list = []
    g_list = []

    for seed_idx in range(n_seeds):
        seed_est = 100 + seed_idx  # different from seed_dgp=42

        # DGP with this R
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

        # Estimation with different seed
        model = ce.Model()
        cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                                "n_covariates": n_cov, "n_simulations": 1},
                 "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
        id_e = {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                "obs_bundles": obs_b}
        item_e = {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars, "beta": beta, "R": R_val, "seed": seed_est}
        model.load_config(cfg_e)
        model.data.load_and_distribute_input_data({"id_data": id_e, "item_data": item_e})
        co2, eo2 = build_oracles(model, seed=seed_est)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(co2)
        model.features.set_error_oracle(eo2)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
        f_list.append(f_val)
        g_list.append(g_val)

    f_arr = np.array(f_list)
    g_arr = np.array(g_list)  # (n_seeds, n_cov)
    mean_g = g_arr.mean(0)
    mean_f = f_arr.mean()
    std_gnorm = np.std([np.linalg.norm(g) for g in g_arr])

    g_str = "  ".join(f"{mean_g[j]:+12.6f}" for j in range(n_cov))
    print(f"  {R_val:>5}  {mean_f:10.4f}  {g_str}  {np.linalg.norm(mean_g):12.6f}  {std_gnorm:10.4f}")

print()
print("If |mean_g| decreases with R: confirms SAA-correlation bias.")
print("If |mean_g| stays constant: bias has a different source.")
