"""Critical test: should DGP and estimation share eps_2?

Theory: V_i2(b_1; theta) = E_{eps_i2}[max_{b_2} ...] is the TRUE expectation.
In the SAA approximation, we fix R draws of eps_2 (the "scenarios").

Hypothesis: the DGP and estimation must use the SAME eps_2 scenarios.
Only eps_1 should differ (simulated fresh for estimation).

When both share eps_2:
  - hat_b_1 is correlated with eps_2 (from DGP's joint MIP)
  - b_1_star is correlated with eps_2 in the SAME WAY (same MIP structure)
  - So E[cov_V - cov_Q] = 0 at theta_star (correlations cancel)

When they use DIFFERENT eps_2:
  - hat_b_1 is correlated with eps_2_dgp (not eps_2_est)
  - b_1_star is correlated with eps_2_est (not eps_2_dgp)
  - The correlations DON'T cancel → systematic bias

Test:
  (A) Current code: different eps_1 AND eps_2 → expect nonzero gradient
  (B) Shared eps_2: different eps_1, SAME eps_2  → expect zero gradient
  (C) Shared eps_1: SAME eps_1, different eps_2  → expect nonzero gradient
"""
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

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

names = ["g_rev", "g_s", "g_c"]


def build_oracles_split(model, seed_eps1, seed_eps2):
    """Build oracles with separate seeds for eps1 and eps2."""
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M_loc = model.config.dimensions.n_items
    R_loc = ld.item_data["R"]
    beta_loc = ld.item_data["beta"]
    rev_c1 = ld.item_data["rev_chars_1"]
    rev_c2 = ld.item_data["rev_chars_2"]
    C = ld.item_data["syn_chars"]

    eps1 = np.zeros((n, M_loc))
    eps2 = np.zeros((n, R_loc, M_loc))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps1[i] = np.random.default_rng((seed_eps1, gid, 0)).normal(0, 1, M_loc)
        eps2[i] = np.random.default_rng((seed_eps2, gid, 1)).normal(0, 1, (R_loc, M_loc))

    ld.errors["eps1"] = eps1
    ld.errors["eps2"] = eps2

    def _get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    def covariates_oracle(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)
        x_rev = b_1 @ rev_c1.T + beta_loc * np.einsum('nrm,km->nk', b_2_r, rev_c2) / R_loc
        x_s = ((1 - b_0) * b_1).sum(-1) + beta_loc * ((1 - b_1)[:, None, :] * b_2_r).sum(-1).mean(-1)
        x_c = np.einsum('nj,jk,nk->n', b_1, C, b_1) + beta_loc * np.einsum('nrj,jk,nrk->n', b_2_r, C, b_2_r) / R_loc
        return np.column_stack([x_rev, x_s, x_c])

    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        e1 = (eps1[ids] * b_1).sum(-1)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)
        e2 = (eps2[ids] * b_2_r).sum(-1).mean(-1)
        return e1 + beta_loc * e2

    return covariates_oracle, error_oracle


# ── DGP ──────────────────────────────────────────────────────────────
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

print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}")
print(f"theta_true={theta_true}")
print(f"Averaging over {n_seeds} estimation seeds")
print()

# ── Test each configuration ──────────────────────────────────────────
configs = {
    "(A) diff eps1 + diff eps2 (current code)": lambda s: (s, s),
    "(B) diff eps1 + SAME eps2 (shared scenarios)": lambda s: (s, seed_dgp),
    "(C) SAME eps1 + diff eps2":                   lambda s: (seed_dgp, s),
}

for label, seed_fn in configs.items():
    g_list = []
    f_list = []
    ndiff_list = []
    for seed_idx in range(n_seeds):
        seed_est = 100 + seed_idx
        s1, s2 = seed_fn(seed_est)

        model = ce.Model()
        cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                                "n_covariates": n_cov, "n_simulations": 1},
                 "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
        model.load_config(cfg_e)
        model.data.load_and_distribute_input_data({
            "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                        "obs_bundles": obs_b},
            "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                          "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_est}})
        co2, eo2 = build_oracles_split(model, seed_eps1=s1, seed_eps2=s2)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(co2)
        model.features.set_error_oracle(eo2)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
        pol = model.data.local_data.id_data["policies"]
        n_diff = (pol["b_1_star"] != obs_b).any(1).sum()
        g_list.append(g_val)
        f_list.append(f_val)
        ndiff_list.append(n_diff)

    g_arr = np.array(g_list)
    f_arr = np.array(f_list)
    mean_g = g_arr.mean(0)
    se_g = g_arr.std(0) / np.sqrt(n_seeds)

    print(f"{label}")
    print(f"  mean_f = {f_arr.mean():.6f} +/- {f_arr.std()/np.sqrt(n_seeds):.4f}")
    print(f"  mean b1_diff = {np.mean(ndiff_list):.1f}/{n_obs}")
    for j, n in enumerate(names):
        t_stat = abs(mean_g[j]) / se_g[j] if se_g[j] > 1e-12 else 0
        print(f"  mean_{n} = {mean_g[j]:+.6f} +/- {se_g[j]:.6f}  (t={t_stat:.2f})")
    print(f"  |mean_g| = {np.linalg.norm(mean_g):.6f}")
    print()
