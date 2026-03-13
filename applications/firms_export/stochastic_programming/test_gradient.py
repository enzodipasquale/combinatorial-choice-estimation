"""Compare gradients across different error draws at several theta points."""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Same setup as test_bundle.py ─────────────────────────────────────
beta = 4
M, K = 20, 20
R_dgp = 1
R_est = 1
S_est = 1
n_obs = 500
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0] * n_rev + [-20.0, 0.1])
seed_dgp = 42
error_seeds = [42, 43, 44, 100, 200]


rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-.1, .1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-.1, .1, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 0.5).astype(float)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars,
                  "beta": beta, "R": R_dgp, "seed": seed_dgp},
}
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M,
                   "n_covariates": n_cov},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

# ── DGP ──────────────────────────────────────────────────────────────
dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
cov_oracle, err_oracle = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))
# ── Eval points ──────────────────────────────────────────────────────
theta_points = {
    "theta_true":       theta_true,
    "theta0":           np.array([0.0, -7.0, 0.0]),
    "midpoint":         theta_true * 0.5 + np.array([0.0, -3.5, 0.0]) * 0.5,
    "near_true":        theta_true + np.array([0.2, 0.5, -0.1]),
}

# ── Error seeds to compare ───────────────────────────────────────────

names = [f"θ_rev{i}" for i in range(n_rev)] + ["θ_s", "θ_c"]
header = f"  {'seed':>6}  {'f':>12}  " + "  ".join(f"{n:>12}" for n in names) + f"  {'|g|':>10}"
sep = "  " + "-" * (len(header) - 2)

is_root = dgp.comm_manager.is_root()

for label, theta in theta_points.items():
    if is_root:
        print(f"\n{'='*70}")
        print(f"  {label}:  theta = {theta}")
        print(f"{'='*70}")
        print(header)
        print(sep)

    for seed_est in error_seeds:
        model = ce.Model()
        cfg_copy = {
            "dimensions": {"n_obs": n_obs, "n_items": M,
                           "n_covariates": n_cov, "n_simulations": S_est},
            "subproblem": {"gurobi_params": {"TimeLimit": 10}},
        }
        input_data_copy = {
            "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                        "obs_bundles": obs_b_dgp},
            "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                          "syn_chars": syn_chars,
                          "beta": beta, "R": R_est, "seed": seed_est},
        }
        model.load_config(cfg_copy)
        model.data.load_and_distribute_input_data(input_data_copy)
        cov_o, err_o = build_oracles(model, seed=seed_est)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_o)
        model.features.set_error_oracle(err_o)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)
        if model.comm_manager.is_root():
            g_str = "  ".join(f"{g_val[j]:+12.4f}" for j in range(n_cov))
            print(f"  {seed_est:>6}  {f_val:12.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
