import numpy as np
import combest as ce
from solver import TwoStageSolverSplit
from oracles import build_oracles

# ── Settings ──────────────────────────────────────────────────────────
beta = .8
M = 10
R_dgp = 200
R_est = 200
S_est = 1
n_obs = 1000
n_rev = 1
n_per_period = n_rev + 3
n_cov = 2 * n_per_period
theta_true = np.array([
    # period 1: rev, s, sc, c
    1.0, -15.0, -2.0, 0.05,
    # period 2: rev, s, sc, c
    1.0, -15.0, -2.0, 0.05,
])
sigma_eps = 0.5
sigma_nu_1 = 1.0
sigma_nu_2 = 1.0
seed_dgp = 42
error_seeds = [seed_dgp, 43, 44, 100, 200]

# ── Draw characteristics ─────────────────────────────────────────────
rng = np.random.default_rng(seed_dgp)
rev_base = rng.uniform(0, 1.0, (n_rev, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (n_obs, n_rev, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (n_obs, n_rev, M))
state_chars = (rng.random((n_obs, M)) > .9).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": beta, "R": R_dgp},
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
cov_oracle, err_oracle = build_oracles(dgp, seed=seed_dgp,
                                       sigma_eps=sigma_eps,
                                       sigma_nu_1=sigma_nu_1,
                                       sigma_nu_2=sigma_nu_2)
dgp.subproblems.load_solver(TwoStageSolverSplit)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(theta_true)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))

# ── Eval ─────────────────────────────────────────────────────────────
names_1 = [f"rev1_{i}" for i in range(n_rev)] + ["s1", "sc1", "c1"]
names_2 = [f"rev2_{i}" for i in range(n_rev)] + ["s2", "sc2", "c2"]
names = names_1 + names_2
header = f"  {'seed':>6}  {'f':>12}  " + "  ".join(f"{n:>12}" for n in names) + f"  {'|g|':>10}"
sep = "  " + "-" * (len(header) - 2)

is_root = dgp.comm_manager.is_root()

if is_root:
    print(f"\n{'='*70}")
    print(f"  theta_true = {theta_true}")
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
        "id_data": {"state_chars": state_chars,
                    "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                    "obs_bundles": obs_b_dgp},
        "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                      "beta": beta, "R": R_est},
    }
    model.load_config(cfg_copy)
    model.data.load_and_distribute_input_data(input_data_copy)
    cov_o, err_o = build_oracles(model, seed=seed_est,
                                 sigma_eps=sigma_eps,
                                 sigma_nu_1=sigma_nu_1,
                                 sigma_nu_2=sigma_nu_2)
    model.subproblems.load_solver(TwoStageSolverSplit)
    model.subproblems.initialize_solver()
    model.features.set_covariates_oracle(cov_o)
    model.features.set_error_oracle(err_o)

    f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta_true)
    if model.comm_manager.is_root():
        g_str = "  ".join(f"{g_val[j]:+12.4f}" for j in range(n_cov))
        print(f"  {seed_est:>6}  {f_val:12.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
