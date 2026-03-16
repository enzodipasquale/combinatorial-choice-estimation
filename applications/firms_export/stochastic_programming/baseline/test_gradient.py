import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

BETA = 0.8
M = 5
R_DGP = 100
R_EST = 100
N_OBS = 200
N_REV = 1
N_COV = N_REV + 3
S_EST = 1

THETA_TRUE = np.array([1] * N_REV + [-15.0, -2.0, 0.05])

SIGMA_1 = 1.0
SIGMA_2 = 5.0

SEED_DGP = 42
ERROR_SEEDS = [SEED_DGP, 43, 44, 100, 200]

rng = np.random.default_rng(SEED_DGP)
rev_base = rng.uniform(0, 1.0, (N_REV, M))
rev_chars_1 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
rev_chars_2 = rev_base[None, :, :] + rng.uniform(0, 1, (N_OBS, N_REV, M))
state_chars = (rng.random((N_OBS, M)) > 0.9).astype(float)
entry_chars = rng.uniform(0, 1, M)
_raw = rng.uniform(0, 1, (M, M))
syn_chars = (_raw + _raw.T) / 2
np.fill_diagonal(syn_chars, 0)

input_data = {
    "id_data": {"state_chars": state_chars,
                "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2},
    "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                  "beta": BETA, "R": R_DGP},
}
cfg = {
    "dimensions": {"n_obs": N_OBS, "n_items": M,
                   "n_covariates": N_COV},
    "subproblem": {"gurobi_params": {"TimeLimit": 10}},
}

dgp = ce.Model()
dgp.load_config(cfg)
dgp.data.load_and_distribute_input_data(input_data)
cov_oracle, err_oracle = build_oracles(dgp, seed=SEED_DGP,
                                       sigma_1=SIGMA_1,
                                       sigma_2=SIGMA_2)
dgp.subproblems.load_solver(TwoStageSolver)
dgp.subproblems.initialize_solver()
dgp.features.set_covariates_oracle(cov_oracle)
dgp.features.set_error_oracle(err_oracle)
obs_b_dgp = dgp.subproblems.generate_obs_bundles(THETA_TRUE)
if dgp.comm_manager.is_root():
    print("Items:", M)
    print(obs_b_dgp.sum(1))
    print(dgp.data.local_data.id_data["policies"]["b_2_r_V"].sum(-1).mean(-1))

theta_points = {
    "theta_true": THETA_TRUE,
}

names = [f"rev{i}" for i in range(N_REV)] + ["entry_c", "entry_dist", "syn"]
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

    for seed_est in ERROR_SEEDS:
        model = ce.Model()
        cfg_copy = {
            "dimensions": {"n_obs": N_OBS, "n_items": M,
                           "n_covariates": N_COV, "n_simulations": S_EST},
            "subproblem": {"gurobi_params": {"TimeLimit": 10}},
        }
        input_data_copy = {
            "id_data": {"state_chars": state_chars,
                        "rev_chars_1": rev_chars_1, "rev_chars_2": rev_chars_2,
                        "obs_bundles": obs_b_dgp},
            "item_data": {"syn_chars": syn_chars, "entry_chars": entry_chars,
                          "beta": BETA, "R": R_EST},
        }
        model.load_config(cfg_copy)
        model.data.load_and_distribute_input_data(input_data_copy)
        cov_o, err_o = build_oracles(model, seed=seed_est,
                                     sigma_1=SIGMA_1,
                                     sigma_2=SIGMA_2)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_o)
        model.features.set_error_oracle(err_o)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)
        if model.comm_manager.is_root():
            g_str = "  ".join(f"{g_val[j]:+12.4f}" for j in range(N_COV))
            print(f"  {seed_est:>6}  {f_val:12.4f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
