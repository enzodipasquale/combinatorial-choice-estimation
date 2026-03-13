"""Detailed investigation of WHY theta_c gradient = 0 when beta=1.

Look at the actual bundles for agents where b_1_star != obs_b.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

M, K_test = 15, 10
R = 30
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
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
C = syn_chars

print(f"M={M}  K={K_test}  R={R}  n_obs={n_obs}  beta={beta}")
print(f"theta_true = {theta_true}")
print()

# ── DGP ──────────────────────────────────────────────────────────────
input_data = {
    "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K_test)},
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

# ── Estimation ───────────────────────────────────────────────────────
model = ce.Model()
cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                        "n_covariates": n_cov, "n_simulations": 1},
         "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
id_e = {"state_chars": state_chars, "capacity": np.full(n_obs, K_test),
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
print(f"f = {f_val}")
print(f"g = {g_val}")
print(f"|g| = {np.linalg.norm(g_val)}")
print()

pol = model.data.local_data.id_data["policies"]
b_1_star = pol["b_1_star"]
b_2_r_V = pol["b_2_r_V"]
b_2_r_Q = pol["b_2_r_Q"]
eps1 = model.data.local_data.errors["eps1"]
eps2 = model.data.local_data.errors["eps2"]

# ── Agents where b_1_star != obs_b ───────────────────────────────────
diff_mask = (b_1_star != obs_b).any(1)
diff_ids = np.where(diff_mask)[0]
print(f"Agents with b_1_star != obs_b: {len(diff_ids)} / {n_obs}")
print(f"Agent indices: {diff_ids}")
print()

for i in diff_ids[:5]:  # show first 5
    b1s = b_1_star[i].astype(int)
    ob = obs_b[i].astype(int)
    flip = b1s != ob

    print(f"{'='*70}")
    print(f"  Agent {i}")
    print(f"  b_1_star = {b1s}  (sum={b1s.sum()})")
    print(f"  obs_b    = {ob}   (sum={ob.sum()})")
    print(f"  flips at items: {np.where(flip)[0]}  "
          f"(in V not Q: {np.where(b1s & ~ob)[0]}, "
          f"in Q not V: {np.where(~b1s.astype(bool) & ob.astype(bool))[0]})")

    # Period 1 synergy
    syn_V1 = b1s @ C @ b1s
    syn_Q1 = ob @ C @ ob
    print(f"  Period 1 synergy: V={syn_V1:.6f}  Q={syn_Q1:.6f}  Δ={syn_V1 - syn_Q1:.6f}")

    # Period 2 synergy (averaged over R)
    b2V = b_2_r_V[i].astype(float)
    b2Q = b_2_r_Q[i].astype(float)
    syn_V2 = np.einsum('rj,jk,rk->r', b2V, C, b2V).mean()
    syn_Q2 = np.einsum('rj,jk,rk->r', b2Q, C, b2Q).mean()
    print(f"  Period 2 synergy: V={syn_V2:.6f}  Q={syn_Q2:.6f}  Δ={syn_V2 - syn_Q2:.6f}")

    # Total synergy covariate
    xc_V = syn_V1 + beta * syn_V2
    xc_Q = syn_Q1 + beta * syn_Q2
    print(f"  Total x_c: V={xc_V:.6f}  Q={xc_Q:.6f}  Δ={xc_V - xc_Q:.10f}")

    # Revenue covariate
    xr_V = b1s @ rev_chars_1.T + beta * np.einsum('rm,km->k', b2V, rev_chars_2) / R
    xr_Q = ob @ rev_chars_1.T + beta * np.einsum('rm,km->k', b2Q, rev_chars_2) / R
    print(f"  Δx_rev = {(xr_V - xr_Q).ravel()}")

    # Entry cost covariate
    xs_V = b1s.sum() + beta * ((1 - b1s) * b2V).sum(-1).mean()
    xs_Q = ob.sum() + beta * ((1 - ob.astype(float)) * b2Q).sum(-1).mean()
    print(f"  Δx_s = {xs_V - xs_Q:.6f}")

    # Check per-scenario b_2 differences
    b2_diff = (b_2_r_V[i] != b_2_r_Q[i])
    n_scen_diff = b2_diff.any(1).sum()
    print(f"  b_2 scenarios differing: {n_scen_diff}/{R}")

    # Show scenario-level synergy
    if n_scen_diff > 0 and n_scen_diff <= 5:
        diff_r = np.where(b2_diff.any(1))[0]
        for r in diff_r[:3]:
            sv = b_2_r_V[i, r].astype(float) @ C @ b_2_r_V[i, r].astype(float)
            sq = b_2_r_Q[i, r].astype(float) @ C @ b_2_r_Q[i, r].astype(float)
            print(f"    r={r}: syn_V2={sv:.4f}  syn_Q2={sq:.4f}  Δ={sv-sq:.6f}")
            print(f"           b2V={b_2_r_V[i,r].astype(int)}")
            print(f"           b2Q={b_2_r_Q[i,r].astype(int)}")
    print()

# ── Manual gradient with higher precision ────────────────────────────
print("=" * 70)
print("MANUAL gradient computation (high precision)")
print("=" * 70)
b1f = b_1_star.astype(float)
obf = obs_b.astype(float)
b2Vf = b_2_r_V.astype(float)
b2Qf = b_2_r_Q.astype(float)

# Per-agent synergy difference
xc_V_all = np.einsum('nj,jk,nk->n', b1f, C, b1f) + beta * np.einsum('nrj,jk,nrk->n', b2Vf, C, b2Vf) / R
xc_Q_all = np.einsum('nj,jk,nk->n', obf, C, obf) + beta * np.einsum('nrj,jk,nrk->n', b2Qf, C, b2Qf) / R
delta_xc = xc_V_all - xc_Q_all

print(f"\nPer-agent Δx_c for differing agents:")
for i in diff_ids:
    print(f"  agent {i}: Δx_c = {delta_xc[i]:.15e}")
print(f"\nOverall mean Δx_c = {delta_xc.mean():.15e}")
print(f"Sum Δx_c = {delta_xc.sum():.15e}")
print()

# ── Now test with different beta values to see when theta_c "turns on" ──
print("=" * 70)
print("Per-agent Δx_c decomposition (period 1 vs period 2)")
print("=" * 70)
delta_p1 = np.einsum('nj,jk,nk->n', b1f, C, b1f) - np.einsum('nj,jk,nk->n', obf, C, obf)
delta_p2_V = np.einsum('nrj,jk,nrk->n', b2Vf, C, b2Vf) / R
delta_p2_Q = np.einsum('nrj,jk,nrk->n', b2Qf, C, b2Qf) / R
delta_p2 = delta_p2_V - delta_p2_Q

for i in diff_ids[:5]:
    print(f"  agent {i}: Δp1 = {delta_p1[i]:.10f}  Δp2 = {delta_p2[i]:.10f}  "
          f"  Δp1 + β*Δp2 = {delta_p1[i] + beta * delta_p2[i]:.10f}")
    print(f"             p2_V = {delta_p2_V[i]:.6f}  p2_Q = {delta_p2_Q[i]:.6f}")
