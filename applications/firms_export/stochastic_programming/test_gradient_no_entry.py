"""Compare gradients across different error draws at several theta points
(no entry cost variant)."""
import numpy as np
import combest as ce
from solver_no_entry import TwoStageNoEntrySolver
from oracles_no_entry import build_oracles

# ── Same setup as test_bundle_no_entry.py ──────────────────────────
beta = 3
M, K = 5, 5
R_dgp = 50
R_est = 50
S_est = 10
n_obs = 100
n_rev = 1
n_syn = 1
n_cov = n_rev + n_syn
theta_true = np.array([1.0] * n_rev + [0.2] * n_syn)
seed_dgp = 42
error_seeds = [seed_dgp, 43, 44, 100, 200]

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(-1.0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
syn_chars = np.zeros((n_syn, M, M))
for l in range(n_syn):
    _raw = rng.uniform(0, 1, (M, M))
    syn_chars[l] = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars[l], 0)

input_data = {
    "id_data": {"capacity": np.full(n_obs, K)},
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
dgp.subproblems.load_solver(TwoStageNoEntrySolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))

# ── Eval points ──────────────────────────────────────────────────────
theta_points = {
    "theta_true":   theta_true,
    "theta0":       np.zeros(n_cov),
    # "midpoint":     theta_true * 0.5,
    "near_true":    theta_true + np.array([.5] * n_rev + [0.005] * n_syn),
}

# ── Error seeds to compare ──────────────────────────────────────────
names = [f"θ_rev{i}" for i in range(n_rev)] + [f"θ_syn{i}" for i in range(n_syn)]
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
            "id_data": {"capacity": np.full(n_obs, K),
                        "obs_bundles": obs_b_dgp},
            "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                          "syn_chars": syn_chars,
                          "beta": beta, "R": R_est, "seed": seed_est},
        }
        model.load_config(cfg_copy)
        model.data.load_and_distribute_input_data(input_data_copy)
        cov_o, err_o = build_oracles(model, seed=seed_est)
        model.subproblems.load_solver(TwoStageNoEntrySolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_o)
        model.features.set_error_oracle(err_o)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)
        if model.comm_manager.is_root():
            g_str = "  ".join(f"{g_val[j]:+12.4f}" for j in range(n_cov))
            print(f"  {seed_est:>6}  {f_val:12.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
