"""Test: does using FRESH eps_2 for covariate evaluation fix the gradient bias?

The hypothesis is that b_1_star from the joint MIP is correlated with eps_2
used for the SAA. If we evaluate covariates with FRESH eps_2 (independent of
the ones used to find b_1_star), the correlation breaks and the gradient
should become unbiased.

Approach:
1. Normal: solve + evaluate covariates with SAME eps_2 (current code)
2. Fresh: solve with eps_2_solve, then RE-SOLVE b_2 with fresh eps_2_eval

We average over seeds to isolate the mean (bias) from variance.
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

# ── Setup ──────────────────────────────────────────────────────────────
beta = 3
M, K = 5, 5
R = 30
n_obs = 200
n_rev = 1
n_cov = n_rev + 2
theta_true = np.array([1.0, -5.0, 0.1])
seed_dgp = 42
n_seeds = 10

rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)
C = syn_chars

# Generate DGP observed bundles ONCE (they don't change)
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

names = ["g_rev", "g_s", "g_c"]


def compute_covariates(b_1, b_2_r):
    """Compute covariates for given b_1 and b_2_r arrays."""
    b_1f = b_1.astype(float)
    b_2f = b_2_r.astype(float)
    x_rev = b_1f @ rev_chars_1.T + beta * np.einsum('nrm,km->nk', b_2f, rev_chars_2) / R
    x_s = b_1f.sum(-1) + beta * ((1 - b_1f)[:, None, :] * b_2f).sum(-1).mean(-1)
    x_c = np.einsum('nj,jk,nk->n', b_1f, C, b_1f) + beta * np.einsum('nrj,jk,nrk->n', b_2f, C, b_2f) / R
    return np.column_stack([x_rev, x_s, x_c])


def reoptimize_b2(b_1, eps2, theta):
    """Given b_1 and fresh eps2, re-solve b_2 per scenario."""
    import gurobipy as gp
    theta_rev = theta[:n_rev]
    theta_s = theta[n_rev]
    theta_c = theta[n_rev + 1]
    rev2 = rev_chars_2.T @ theta_rev

    n = b_1.shape[0]
    b_2_r = np.zeros((n, R, M), dtype=bool)

    for i in range(n):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.TimeLimit = 10
        b2 = m.addMVar(M, vtype=gp.GRB.BINARY, name='b_2')
        m.addConstr(b2.sum() <= K)
        m.update()
        for r in range(R):
            c = rev2 + (1 - b_1[i].astype(float)) * theta_s + eps2[i, r]
            m.setObjective(c @ b2 + theta_c * (b2 @ C @ b2), gp.GRB.MAXIMIZE)
            m.optimize()
            b_2_r[i, r] = np.array(b2.X) > 0.5
        m.close()
        env.close()
    return b_2_r


print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}")
print(f"theta_true={theta_true}  DGP seed={seed_dgp}")
print(f"Averaging over {n_seeds} estimation seeds")
print()

g_same_list = []
g_fresh_list = []
f_same_list = []
f_fresh_list = []

for seed_idx in range(n_seeds):
    seed_est = 100 + seed_idx

    # ── Standard estimation (SAME eps_2 for solve + eval) ──
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
    g_same_list.append(g_val)
    f_same_list.append(f_val)

    # Get the policies and eps2 from this run
    pol = model.data.local_data.id_data["policies"]
    b_1_star = pol["b_1_star"].copy()
    eps2_est = model.data.local_data.errors["eps2"]

    # ── Fresh evaluation: use different eps_2 for covariates ──
    seed_fresh = 5000 + seed_idx
    n_agents = n_obs
    eps2_fresh = np.zeros((n_agents, R, M))
    for i in range(n_agents):
        eps2_fresh[i] = np.random.default_rng((seed_fresh, i, 1)).normal(0, 1, (R, M))

    # Re-solve b_2 for V with fresh eps_2
    b_2_r_V_fresh = reoptimize_b2(b_1_star, eps2_fresh, theta_true)
    # Re-solve b_2 for Q with fresh eps_2
    b_2_r_Q_fresh = reoptimize_b2(obs_b, eps2_fresh, theta_true)

    cov_V_fresh = compute_covariates(b_1_star, b_2_r_V_fresh)
    cov_Q_fresh = compute_covariates(obs_b, b_2_r_Q_fresh)
    g_fresh = (cov_V_fresh - cov_Q_fresh).mean(0)
    f_fresh = (g_fresh @ theta_true)  # approximate (ignoring error terms)
    g_fresh_list.append(g_fresh)
    f_fresh_list.append(f_fresh)

    print(f"  seed {seed_est}: |g_same|={np.linalg.norm(g_val):.4f}  "
          f"|g_fresh|={np.linalg.norm(g_fresh):.4f}")

g_same = np.array(g_same_list)
g_fresh = np.array(g_fresh_list)

print()
print("=" * 70)
print("SUMMARY: Mean gradient over seeds")
print("=" * 70)
print(f"{'':>12}  " + "  ".join(f"{'mean_'+n:>12}" for n in names) + f"  {'|mean_g|':>12}")
print("-" * 80)
mean_same = g_same.mean(0)
mean_fresh = g_fresh.mean(0)
s_same = "  ".join(f"{mean_same[j]:+12.6f}" for j in range(n_cov))
s_fresh = "  ".join(f"{mean_fresh[j]:+12.6f}" for j in range(n_cov))
print(f"  {'SAME eps2':>12}  {s_same}  {np.linalg.norm(mean_same):12.6f}")
print(f"  {'FRESH eps2':>12}  {s_fresh}  {np.linalg.norm(mean_fresh):12.6f}")
print()
print("If FRESH eps2 has |mean_g| ~ 0: confirms the correlation is the sole cause.")
print("If FRESH eps2 still has |mean_g| >> 0: there's another source of bias.")
