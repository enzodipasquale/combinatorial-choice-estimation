"""Find good theta_true for beta=1 with meaningful gradient behavior.

Requirements:
- Non-trivial bundles (avg size 3-8, not 0 or M)
- Good variation (many agents differ between V and Q)
- All gradient components nonzero
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

beta = 1.0
R = 30
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
seed_dgp = 42
seed_est = 43

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, 15))  # generate for M=15, use subset
rev_chars_1_full = rev_base + rng.uniform(-0.1, 0.1, (n_rev, 15))
rev_chars_2_full = rev_base + rng.uniform(-0.1, 0.1, (n_rev, 15))
_raw_full = rng.uniform(0, 1, (15, 15))
syn_chars_full = (_raw_full + _raw_full.T) / 2
np.fill_diagonal(syn_chars_full, 0)

names = ["θ_rev", "θ_s", "θ_c"]

# ── Sweep: (M, K, theta_s, theta_c) ─────────────────────────────────
configs = [
    # (M, K, theta_true)
    (10, 8,  [1.0, -1.0, 0.1]),
    (10, 8,  [1.0, -0.5, 0.1]),
    (10, 8,  [1.0, -1.0, 0.5]),
    (10, 6,  [1.0, -1.0, 0.1]),
    (10, 5,  [1.0, -2.0, 0.1]),
    (15, 10, [1.0, -1.0, 0.1]),
    (15, 10, [1.0, -0.5, 0.5]),
    (10, 8,  [1.0, -2.0, 0.1]),
]

print(f"beta={beta}  R={R}  n_obs={n_obs}")
print()
hdr = (f"  {'M':>3} {'K':>3}  {'theta':>20}  {'f':>8}  "
       + "  ".join(f"{n:>8}" for n in names)
       + f"  {'|g|':>8}  {'diff':>5}  {'avg_sz':>6}")
print(hdr)
print("-" * len(hdr))

for M, K, theta_list in configs:
    theta_true = np.array(theta_list)
    rev_chars_1 = rev_chars_1_full[:, :M]
    rev_chars_2 = rev_chars_2_full[:, :M]
    syn_chars = syn_chars_full[:M, :M]
    state_chars = np.zeros((n_obs, M))

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

    g_str = "  ".join(f"{g_val[j]:+8.4f}" for j in range(n_cov))
    theta_str = str(theta_true)
    print(f"  {M:>3} {K:>3}  {theta_str:>20}  {f_val:8.4f}  {g_str}  {np.linalg.norm(g_val):8.4f}"
          f"  {n_diff:>5}  {avg_sz:6.1f}")
