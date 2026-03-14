"""Definitive population bias test.

Previous test was misleading: varying est_seed but keeping DGP fixed gives
a nonzero mean gradient that reflects the specific DGP realization, not
the population bias.

This test varies BOTH DGP seed and estimation seed, so dataset-specific
effects average out. What remains is the true population bias.

Configurations:
  (A) diff eps1 + diff eps2 (current code)
  (B) diff eps1 + SAME eps2 (shared scenarios)
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
n_experiments = 20  # each with different DGP + estimation seed

rng_setup = np.random.default_rng(42)
rev_base = rng_setup.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base + rng_setup.uniform(-0.1, 0.1, (n_rev, M))
rev_chars_2 = rev_base + rng_setup.uniform(-0.1, 0.1, (n_rev, M))
state_chars = np.zeros((n_obs, M))
_raw = rng_setup.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

names = ["g_rev", "g_s", "g_c"]


def build_oracles_split(model, seed_eps1, seed_eps2):
    """Build oracles with separate seeds for eps1 and eps2."""
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    Ml = model.config.dimensions.n_items
    Rl = ld.item_data["R"]
    bl = ld.item_data["beta"]
    rc1 = ld.item_data["rev_chars_1"]
    rc2 = ld.item_data["rev_chars_2"]
    C = ld.item_data["syn_chars"]

    eps1 = np.zeros((n, Ml))
    eps2 = np.zeros((n, Rl, Ml))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps1[i] = np.random.default_rng((seed_eps1, gid, 0)).normal(0, 1, Ml)
        eps2[i] = np.random.default_rng((seed_eps2, gid, 1)).normal(0, 1, (Rl, Ml))
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
        x_rev = b_1 @ rc1.T + bl * np.einsum('nrm,km->nk', b_2_r, rc2) / Rl
        x_s = ((1 - b_0) * b_1).sum(-1) + bl * ((1 - b_1)[:, None, :] * b_2_r).sum(-1).mean(-1)
        x_c = np.einsum('nj,jk,nk->n', b_1, C, b_1) + bl * np.einsum('nrj,jk,nrk->n', b_2_r, C, b_2_r) / Rl
        return np.column_stack([x_rev, x_s, x_c])

    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        e1 = (eps1[ids] * b_1).sum(-1)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)
        e2 = (eps2[ids] * b_2_r).sum(-1).mean(-1)
        return e1 + bl * e2

    return covariates_oracle, error_oracle


print(f"M={M}  K={K}  R={R}  n_obs={n_obs}  beta={beta}")
print(f"theta_true={theta_true}")
print(f"Running {n_experiments} experiments (each with different DGP + est seed)")
print()

g_A_list = []
g_B_list = []
f_A_list = []
f_B_list = []

for exp_idx in range(n_experiments):
    seed_dgp = 42 + exp_idx
    seed_est = 1000 + exp_idx

    # DGP
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

    # --- (A) diff eps1 + diff eps2 ---
    model_A = ce.Model()
    cfg_e = {"dimensions": {"n_obs": n_obs, "n_items": M,
                            "n_covariates": n_cov, "n_simulations": 1},
             "subproblem": {"gurobi_params": {"TimeLimit": 10}}}
    model_A.load_config(cfg_e)
    model_A.data.load_and_distribute_input_data({
        "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                    "obs_bundles": obs_b},
        "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                      "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_est}})
    co_A, eo_A = build_oracles_split(model_A, seed_eps1=seed_est, seed_eps2=seed_est)
    model_A.subproblems.load_solver(TwoStageSolver)
    model_A.subproblems.initialize_solver()
    model_A.features.set_covariates_oracle(co_A)
    model_A.features.set_error_oracle(eo_A)
    f_A, g_A = model_A.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    g_A_list.append(g_A)
    f_A_list.append(f_A)

    # --- (B) diff eps1 + SAME eps2 ---
    model_B = ce.Model()
    model_B.load_config(cfg_e)
    model_B.data.load_and_distribute_input_data({
        "id_data": {"state_chars": state_chars, "capacity": np.full(n_obs, K),
                    "obs_bundles": obs_b},
        "item_data": {"rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                      "syn_chars": syn_chars, "beta": beta, "R": R, "seed": seed_est}})
    co_B, eo_B = build_oracles_split(model_B, seed_eps1=seed_est, seed_eps2=seed_dgp)
    model_B.subproblems.load_solver(TwoStageSolver)
    model_B.subproblems.initialize_solver()
    model_B.features.set_covariates_oracle(co_B)
    model_B.features.set_error_oracle(eo_B)
    f_B, g_B = model_B.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    g_B_list.append(g_B)
    f_B_list.append(f_B)

    print(f"  exp {exp_idx}: (A) |g|={np.linalg.norm(g_A):.4f}  "
          f"(B) |g|={np.linalg.norm(g_B):.4f}")

g_A_arr = np.array(g_A_list)
g_B_arr = np.array(g_B_list)
f_A_arr = np.array(f_A_list)
f_B_arr = np.array(f_B_list)

print()
print("=" * 70)
print("POPULATION BIAS (averaged over DGP AND estimation seeds)")
print("=" * 70)

for label, g_arr, f_arr in [("(A) diff eps1+eps2", g_A_arr, f_A_arr),
                              ("(B) diff eps1, SAME eps2", g_B_arr, f_B_arr)]:
    mean_g = g_arr.mean(0)
    se_g = g_arr.std(0) / np.sqrt(n_experiments)
    mean_f = f_arr.mean()
    se_f = f_arr.std() / np.sqrt(n_experiments)
    print(f"\n{label}:")
    print(f"  mean_f = {mean_f:.6f} +/- {se_f:.4f}")
    for j, n in enumerate(names):
        t = abs(mean_g[j]) / se_g[j] if se_g[j] > 1e-12 else 0
        print(f"  mean_{n} = {mean_g[j]:+.6f} +/- {se_g[j]:.6f}  (t={t:.2f})")
    print(f"  |mean_g| = {np.linalg.norm(mean_g):.6f}")

print()
print("Interpretation:")
print("  (A) has nonzero |mean_g| → population bias from different eps_2 (SAA mismatch)")
print("  (B) has |mean_g| ≈ 0    → no population bias when eps_2 shared")
print("  If BOTH have |mean_g| ≈ 0, the previous test's 'bias' was just fixed-dataset noise")
