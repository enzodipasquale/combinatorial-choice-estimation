"""Diagnose: why is theta_c gradient always 0 when beta=1?

Tests whether M=K makes the problem trivial (all items selected),
and whether theta_c gradient has structural issues at beta=1.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Same as test_gradient.py but with beta parameter sweep ──────────
M, K = 15, 15
R = 30
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
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["θ_rev", "θ_s", "θ_c"]

print(f"M={M}  K={K}  R={R}  n_obs={n_obs}")
print(f"theta_true = {theta_true}")
print()

# ── TEST 1: beta sweep with M=K=15 (potentially trivial) ────────────
print("=" * 80)
print("TEST 1: beta sweep with M=K=15 (is K=M trivial?)")
print("=" * 80)
hdr = (f"  {'beta':>6}  {'f':>10}  " + "  ".join(f"{n:>10}" for n in names)
       + f"  {'|g|':>10}  {'b1=obs':>8}  {'all1':>8}")
print(hdr)
print("-" * len(hdr))

for beta in [0.5, 1.0, 2.0, 3.0]:
    input_data = {
        "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
        "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                      "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp},
    }
    cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
           "subproblem": {"gurobi_params": {"TimeLimit": 10}}}

    dgp = ce.Model()
    dgp.load_config(cfg)
    dgp.data.load_and_distribute_input_data(input_data)
    co, eo = build_oracles(dgp, seed=seed_dgp)
    dgp.subproblems.load_solver(TwoStageSolver)
    dgp.subproblems.initialize_solver()
    dgp.features.set_covariates_oracle(co)
    dgp.features.set_error_oracle(eo)
    obs_b = dgp.subproblems.generate_obs_bundles(theta_true)

    obs_all1 = (obs_b.sum(1) == M).all()

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
    b1eq = (pol["b_1_star"] == obs_b).all()
    b1_all1 = (pol["b_1_star"].sum(1) == M).all()

    g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
    print(f"  {beta:6.1f}  {f_val:10.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}"
          f"  {str(b1eq):>8}  {str(b1_all1):>8}")


# ── TEST 2: K < M with beta=1 (non-trivial) ─────────────────────────
print()
print("=" * 80)
print("TEST 2: beta=1 with K < M (non-trivial constraint)")
print("=" * 80)

for K_test in [5, 8, 10, 12]:
    beta = 1.0
    input_data = {
        "id_data": {"state_chars": state_chars[:n_obs], "capacity": np.full(n_obs, K_test)},
        "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                      "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp},
    }
    cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
           "subproblem": {"gurobi_params": {"TimeLimit": 10}}}

    dgp = ce.Model()
    dgp.load_config(cfg)
    dgp.data.load_and_distribute_input_data(input_data)
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
    id_e = {"state_chars": state_chars[:n_obs], "capacity": np.full(n_obs, K_test),
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
    b1eq = (pol["b_1_star"] == obs_b).all()
    n_diff = (pol["b_1_star"] != obs_b).any(1).sum()
    b2V = pol["b_2_r_V"]
    b2Q = pol["b_2_r_Q"]
    b2eq = (b2V == b2Q).all()

    g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
    hdr2 = (f"  {'K':>4}  {'f':>10}  " + "  ".join(f"{n:>10}" for n in names)
            + f"  {'|g|':>10}  {'b1_diff':>8}  {'b2_eq':>8}")
    if K_test == 5:
        print(hdr2)
        print("-" * len(hdr2))
    print(f"  {K_test:>4}  {f_val:10.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}"
          f"  {n_diff:>8}  {str(b2eq):>8}")


# ── TEST 3: multiple seeds at beta=1, K=10 ──────────────────────────
print()
print("=" * 80)
print("TEST 3: beta=1, K=10, multiple seeds")
print("=" * 80)
K_test = 10
beta = 1.0
input_data = {
    "id_data": {"state_chars": state_chars[:n_obs], "capacity": np.full(n_obs, K_test)},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp},
}
cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
       "subproblem": {"gurobi_params": {"TimeLimit": 10}}}

dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
co, eo = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(co)
dgp.features.set_error_oracle(eo)
obs_b = dgp.subproblems.generate_obs_bundles(theta_true)

hdr3 = (f"  {'seed':>6}  {'f':>10}  " + "  ".join(f"{n:>10}" for n in names)
        + f"  {'|g|':>10}  {'b1_diff':>8}")
print(hdr3)
print("-" * len(hdr3))

for s in [42, 43, 44, 100, 200]:
    model = ce.Model()
    cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                            "n_covariates": n_cov, "n_simulations": 1},
             "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
    id_e = {"state_chars": state_chars[:n_obs], "capacity": np.full(n_obs, K_test),
            "obs_bundles": obs_b}
    item_e = {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
              "syn_chars": syn_chars, "beta": beta, "R": R, "seed": s}
    model.load_config(cfg_e)
    model.data.load_and_distribute_input_data({"id_data": id_e, "item_data": item_e})
    co2, eo2 = build_oracles(model, seed=s)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(co2)
    model.features.set_error_oracle(eo2)

    f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    pol = model.data.local_data.id_data["policies"]
    n_diff = (pol["b_1_star"] != obs_b).any(1).sum()

    g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
    print(f"  {s:>6}  {f_val:10.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}"
          f"  {n_diff:>8}")
