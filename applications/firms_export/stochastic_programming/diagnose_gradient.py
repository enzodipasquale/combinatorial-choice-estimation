"""Diagnose gradient variance across error seeds (entry-cost case).

Decomposes into:
 - period 1 vs period 2 contributions
 - per-agent variance → expected SE of gradient mean
 - policy flip rates
 - V-covariates vs Q-covariates magnitudes
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Match test_gradient.py settings ────────────────────────────────
beta = 3
M, K = 3, 3
R = 100         # change to compare R=20 vs R=100
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.5])
seed_dgp = 42
seeds = [42, 43, 44, 100, 200]

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = (rng.random((n_obs, M)) > 1).astype(float)  # all zeros
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}")
print(f"theta_true = {theta_true}")
print(f"rev_chars_1 = {rev_chars_1.ravel()}")
print(f"rev_chars_2 = {rev_chars_2.ravel()}")
print(f"state_chars: all zeros = {(state_chars == 0).all()}")
print(f"syn_chars:\n{syn_chars}")
print()

# ── DGP ────────────────────────────────────────────────────────────
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
cov_o, err_o = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_o)
dgp.features.set_error_oracle(err_o)
obs_b = dgp.subproblems.generate_obs_bundles(theta_true)
obs_b_f = obs_b.astype(float)

print(f"obs_b bundle sizes: min={obs_b.sum(1).min()} max={obs_b.sum(1).max()} "
      f"mean={obs_b.sum(1).mean():.2f}")
print(f"obs_b per-item selection rate: {obs_b.mean(0)}")
print()

# ── Helper: manual per-agent covariate computation ─────────────────
def compute_covariates(b_1, b_2_r, b_0):
    b_1f = b_1.astype(float)
    b_2f = b_2_r.astype(float)
    x_rev = b_1f @ rev_chars_1.T + beta * np.einsum('nrm,km->nk', b_2f, rev_chars_2) / R
    x_s = ((1 - b_0) * b_1f).sum(-1) + beta * ((1 - b_1f)[:, None, :] * b_2f).sum(-1).mean(-1)
    x_c = (np.einsum('nj,jk,nk->n', b_1f, syn_chars, b_1f)
            + beta * np.einsum('nrj,jk,nrk->n', b_2f, syn_chars, b_2f) / R)
    return np.column_stack([x_rev, x_s, x_c])

def compute_covariates_by_period(b_1, b_2_r, b_0):
    """Return period-1-only and period-2-only covariates separately."""
    b_1f = b_1.astype(float)
    b_2f = b_2_r.astype(float)
    # Period 1 only
    p1_rev = b_1f @ rev_chars_1.T
    p1_s = ((1 - b_0) * b_1f).sum(-1)
    p1_c = np.einsum('nj,jk,nk->n', b_1f, syn_chars, b_1f)
    # Period 2 only
    p2_rev = beta * np.einsum('nrm,km->nk', b_2f, rev_chars_2) / R
    p2_s = beta * ((1 - b_1f)[:, None, :] * b_2f).sum(-1).mean(-1)
    p2_c = beta * np.einsum('nrj,jk,nrk->n', b_2f, syn_chars, b_2f) / R
    return (np.column_stack([p1_rev, p1_s, p1_c]),
            np.column_stack([p2_rev, p2_s, p2_c]))


# ── Analyze each seed ──────────────────────────────────────────────
all_grads = []
all_grads_p1 = []
all_grads_p2 = []
cov_names = ["θ_rev", "θ_s", "θ_c"]

for seed_est in seeds:
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
    co, eo = build_oracles(model, seed=seed_est)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(co)
    model.features.set_error_oracle(eo)

    f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)

    # Extract policies
    pol = model.data.local_data.id_data["policies"]
    b_1_star = pol["b_1_star"]
    b_2_r_V = pol["b_2_r_V"]
    b_2_r_Q = pol["b_2_r_Q"]

    # Per-agent covariates
    cov_V = compute_covariates(b_1_star, b_2_r_V, state_chars)
    cov_Q = compute_covariates(obs_b, b_2_r_Q, state_chars)
    delta = cov_V - cov_Q  # (n_obs, n_cov) per-agent gradient contrib

    # Period decomposition
    p1_V, p2_V = compute_covariates_by_period(b_1_star, b_2_r_V, state_chars)
    p1_Q, p2_Q = compute_covariates_by_period(obs_b, b_2_r_Q, state_chars)
    delta_p1 = p1_V - p1_Q
    delta_p2 = p2_V - p2_Q

    grad_full = delta.mean(0)
    grad_p1 = delta_p1.mean(0)
    grad_p2 = delta_p2.mean(0)
    all_grads.append(grad_full)
    all_grads_p1.append(grad_p1)
    all_grads_p2.append(grad_p2)

    # Policy analysis
    b1_diff = (b_1_star != obs_b)
    n_agents_diff = b1_diff.any(1).sum()
    per_item_flip = b1_diff.mean(0)

    b2V_vs_Q = (b_2_r_V != b_2_r_Q)
    n_b2_diff = b2V_vs_Q.any(axis=(1, 2)).sum()

    # b_1_star stats
    b1_rate = b_1_star.astype(float).mean(0)

    print(f"{'='*70}")
    print(f"  seed={seed_est}   f={f_val:.4f}   grad={g_val}   |g|={np.linalg.norm(g_val):.4f}")
    print(f"  b_1_star selection rate per item: {b1_rate}")
    print(f"  b_1_star differs from obs_b: {n_agents_diff}/{n_obs} agents")
    print(f"  per-item flip rate: {per_item_flip}")
    print(f"  b_2_r_V != b_2_r_Q: {n_b2_diff}/{n_obs} agents")
    print()
    print(f"  GRADIENT DECOMPOSITION (mean over agents):")
    print(f"  {'':>12} {'full':>12} {'period1':>12} {'period2':>12}")
    for k in range(n_cov):
        print(f"  {cov_names[k]:>12} {grad_full[k]:+12.4f} {grad_p1[k]:+12.4f} {grad_p2[k]:+12.4f}")
    print()
    print(f"  COVARIATE MAGNITUDES (mean over agents):")
    print(f"  {'':>12} {'mean_V':>12} {'mean_Q':>12} {'mean_Δ':>12} {'std_Δ':>12} {'SE':>12}")
    for k in range(n_cov):
        print(f"  {cov_names[k]:>12} {cov_V[:, k].mean():+12.4f} {cov_Q[:, k].mean():+12.4f} "
              f"{delta[:, k].mean():+12.4f} {delta[:, k].std():12.4f} "
              f"{delta[:, k].std() / np.sqrt(n_obs):12.6f}")
    print()

# ── Cross-seed summary ────────────────────────────────────────────
grads = np.array(all_grads)
grads_p1 = np.array(all_grads_p1)
grads_p2 = np.array(all_grads_p2)
non_dgp = grads[1:]  # exclude seed 42
non_dgp_p1 = grads_p1[1:]
non_dgp_p2 = grads_p2[1:]

print(f"\n{'='*70}")
print("  CROSS-SEED SUMMARY (seeds 43, 44, 100, 200)")
print(f"{'='*70}")
print(f"  {'':>12} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
for k in range(n_cov):
    print(f"  grad_{cov_names[k]:>6} {non_dgp[:, k].mean():+12.4f} {non_dgp[:, k].std():12.4f} "
          f"{non_dgp[:, k].min():+12.4f} {non_dgp[:, k].max():+12.4f}")
print()
for k in range(n_cov):
    print(f"  p1_{cov_names[k]:>8} {non_dgp_p1[:, k].mean():+12.4f} {non_dgp_p1[:, k].std():12.4f} "
          f"{non_dgp_p1[:, k].min():+12.4f} {non_dgp_p1[:, k].max():+12.4f}")
print()
for k in range(n_cov):
    print(f"  p2_{cov_names[k]:>8} {non_dgp_p2[:, k].mean():+12.4f} {non_dgp_p2[:, k].std():12.4f} "
          f"{non_dgp_p2[:, k].min():+12.4f} {non_dgp_p2[:, k].max():+12.4f}")

# Compare expected SE (from per-agent variance) with observed cross-seed std
print(f"\n  EXPECTED SE vs OBSERVED STD:")
# Use seed 43 as representative for per-agent variance
model2 = ce.Model()
model2.load_config({"dimensions": {"n_obs": n_obs, "n_items": M,
                                   "n_covariates": n_cov, "n_simulations": 1},
                    "subproblem": {"gurobi_params": {"TimeLimit": 10}}})
model2.data.load_and_distribute_input_data({
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                "obs_bundles": obs_b},
    "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                  "syn_chars": syn_chars, "beta": beta, "R": R, "seed": 43}})
co2, eo2 = build_oracles(model2, seed=43)
model2.subproblems.load_solver(TwoStageSolver)
model2.subproblems.initialize_solver()
model2.features.set_covariates_oracle(co2)
model2.features.set_error_oracle(eo2)
model2.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
pol2 = model2.data.local_data.id_data["policies"]
cov_V2 = compute_covariates(pol2["b_1_star"], pol2["b_2_r_V"], state_chars)
cov_Q2 = compute_covariates(obs_b, pol2["b_2_r_Q"], state_chars)
delta2 = cov_V2 - cov_Q2
print(f"  {'':>12} {'per-agent std':>14} {'SE=std/√N':>14} {'cross-seed std':>14} {'ratio':>10}")
for k in range(n_cov):
    se = delta2[:, k].std() / np.sqrt(n_obs)
    cs = non_dgp[:, k].std()
    print(f"  {cov_names[k]:>12} {delta2[:, k].std():14.4f} {se:14.6f} {cs:14.6f} {cs/se if se > 0 else float('inf'):10.2f}")
