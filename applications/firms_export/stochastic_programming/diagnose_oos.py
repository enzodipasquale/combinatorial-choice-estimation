"""Test hypothesis: gradient bias from in-sample ε₂ in joint MIP.

b_1_star from the joint MIP is correlated with ε₂ (it was optimized using
those exact draws). The oracle then evaluates second-stage covariates with
the SAME ε₂, creating systematic upward bias in V-covariates.

Fix: re-optimize second stage and evaluate with FRESH ε₂ draws (out-of-sample).
b_1_star is then uncorrelated with ε₂_fresh, removing the bias.
"""
import numpy as np
import gurobipy as gp
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

beta = 3
M, K = 10, 10
R = 30
n_obs = 100
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
seeds = [43, 44, 100]

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)
C = syn_chars
theta_rev = theta_true[:n_rev]
theta_s = theta_true[n_rev]
theta_c = theta_true[n_rev + 1]
rev2 = rev_chars_2.T @ theta_rev  # (M,)

# ── DGP ────────────────────────────────────────────────────────────
inp = {"id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K)},
       "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                     "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_dgp}}
cfg = {"dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov},
       "subproblem": {"gurobi_params": {"TimeLimit": 30}}}
dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(inp)
co, eo = build_oracles(dgp, seed=seed_dgp)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(co)
dgp.features.set_error_oracle(eo)
obs_b = dgp.subproblems.generate_obs_bundles(theta_true)
obs_b_f = obs_b.astype(float)


def reopt_b2(b_1, eps2_fresh, cap):
    """Re-optimize second stage with fresh errors."""
    n, R_f, M_f = eps2_fresh.shape
    b_2_r = np.zeros((n, R_f, M_f), dtype=bool)
    for i in range(n):
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        b_2 = m.addMVar(M_f, vtype=gp.GRB.BINARY, name='b_2')
        m.addConstr(b_2.sum() <= cap[i])
        m.update()
        for r in range(R_f):
            c_r = rev2 + (1 - b_1[i].astype(float)) * theta_s + eps2_fresh[i, r]
            m.setObjective(c_r @ b_2 + theta_c * (b_2 @ C @ b_2), gp.GRB.MAXIMIZE)
            m.optimize()
            b_2_r[i, r] = np.array(b_2.X) > 0.5
    return b_2_r


def compute_grad(b_1_star, b_2_r_V, obs_b, b_2_r_Q, eps1, eps2):
    """Manually compute gradient = mean(cov_V - cov_Q)."""
    b1f = b_1_star.astype(float)
    b2Vf = b_2_r_V.astype(float)
    obf = obs_b.astype(float)
    b2Qf = b_2_r_Q.astype(float)
    b0 = state_chars

    # Covariates V
    xr_V = b1f @ rev_chars_1.T + beta * np.einsum('nrm,km->nk', b2Vf, rev_chars_2) / R
    xs_V = ((1 - b0) * b1f).sum(-1) + beta * ((1 - b1f)[:, None, :] * b2Vf).sum(-1).mean(-1)
    xc_V = (np.einsum('nj,jk,nk->n', b1f, C, b1f)
            + beta * np.einsum('nrj,jk,nrk->n', b2Vf, C, b2Vf) / R)

    # Covariates Q
    xr_Q = obf @ rev_chars_1.T + beta * np.einsum('nrm,km->nk', b2Qf, rev_chars_2) / R
    xs_Q = ((1 - b0) * obf).sum(-1) + beta * ((1 - obf)[:, None, :] * b2Qf).sum(-1).mean(-1)
    xc_Q = (np.einsum('nj,jk,nk->n', obf, C, obf)
            + beta * np.einsum('nrj,jk,nrk->n', b2Qf, C, b2Qf) / R)

    # Errors
    e_V = (eps1 * b1f).sum(-1) + beta * (eps2 * b2Vf).sum(-1).mean(-1)
    e_Q = (eps1 * obf).sum(-1) + beta * (eps2 * b2Qf).sum(-1).mean(-1)

    cov_V = np.column_stack([xr_V, xs_V, xc_V])
    cov_Q = np.column_stack([xr_Q, xs_Q, xc_Q])
    grad = (cov_V - cov_Q).mean(0)
    const = (e_V - e_Q).mean()
    f_val = grad @ theta_true + const
    return f_val, grad


print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}  theta={theta_true}")
print()
names = ["θ_rev", "θ_s", "θ_c"]
hdr = f"  {'seed':>5}  {'method':>12}  {'f':>8}  " + "  ".join(f"{n:>10}" for n in names) + f"  {'|g|':>8}"
sep = "-" * len(hdr)

cap = np.full(n_obs, K)

for seed_est in seeds:
    print(f"\n  seed={seed_est}")
    print(hdr)
    print(sep)

    # ── Standard (in-sample) ───────────────────────────────────────
    model = ce.Model()
    cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                            "n_covariates": n_cov, "n_simulations": 1},
             "subproblem": {"gurobi_params": {"TimeLimit": 30}}}
    id_e = {"state_chars": state_chars, "capacity": cap, "obs_bundles": obs_b}
    item_e = {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
              "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_est}
    model.load_config(cfg_e)
    model.data.load_and_distribute_input_data({"id_data": id_e, "item_data": item_e})
    co2, eo2 = build_oracles(model, seed=seed_est)
    model.subproblems.load_solver(TwoStageSolver)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(co2)
    model.features.set_error_oracle(eo2)

    f_is, g_is = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    g_str = "  ".join(f"{g_is[j]:+10.4f}" for j in range(n_cov))
    print(f"  {seed_est:>5}  {'in-sample':>12}  {f_is:8.4f}  {g_str}  {np.linalg.norm(g_is):8.4f}")

    # Get policies from solver
    pol = model.data.local_data.id_data["policies"]
    b_1_star = pol["b_1_star"]
    eps1 = model.data.local_data.errors["eps1"]
    eps2 = model.data.local_data.errors["eps2"]

    # ── Out-of-sample: re-optimize b2 with FRESH eps2 ─────────────
    eps2_fresh = np.zeros_like(eps2)
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps2_fresh[i] = np.random.default_rng((seed_est, gid, 99)).normal(0, 1, (R, M))

    b2_V_fresh = reopt_b2(b_1_star, eps2_fresh, cap)
    b2_Q_fresh = reopt_b2(obs_b, eps2_fresh, cap)

    f_oos, g_oos = compute_grad(b_1_star, b2_V_fresh, obs_b, b2_Q_fresh,
                                 eps1, eps2_fresh)
    g_str = "  ".join(f"{g_oos[j]:+10.4f}" for j in range(n_cov))
    print(f"  {seed_est:>5}  {'out-of-sample':>12}  {f_oos:8.4f}  {g_str}  {np.linalg.norm(g_oos):8.4f}")

    # ── Verify: use SAME eps2 (should match in-sample) ────────────
    f_chk, g_chk = compute_grad(b_1_star, pol["b_2_r_V"], obs_b, pol["b_2_r_Q"],
                                 eps1, eps2)
    g_str = "  ".join(f"{g_chk[j]:+10.4f}" for j in range(n_cov))
    print(f"  {seed_est:>5}  {'verify-IS':>12}  {f_chk:8.4f}  {g_str}  {np.linalg.norm(g_chk):8.4f}")

print()
print("If 'out-of-sample' gradients are smaller/less biased than 'in-sample',")
print("the hypothesis is confirmed: b_1_star is overfitting to eps2.")
