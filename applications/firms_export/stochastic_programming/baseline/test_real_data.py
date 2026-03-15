import sys
from pathlib import Path
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_data_sp import main as load_data, build_input_data_sp

# ── Settings ──────────────────────────────────────────────────────────
COUNTRY = "MEX"
KEEP_TOP = 10
BETA = 0.85
END_BUFFER = 3
R = 100
N_SAMPLE = 500
SIGMA_EPS = 1.0
SIGMA_NU_1 = 0.5
SIGMA_NU_2 = 2
N_SIMULATIONS = 1

thetas = [
    np.zeros(4),
    np.array([0.5, -5.0, -1.0, 0.1]),
    np.array([1.0, -10.0, -2.0, 0.05]),
    np.array([0.1, -1.0, -0.5, 0.0]),
]
error_seeds = [42, 43, 44, 100, 200]

# ── Setup ─────────────────────────────────────────────────────────────
ctx = load_data(COUNTRY, KEEP_TOP, beta=BETA, end_buffer=END_BUFFER,
                n_sample=N_SAMPLE)
M = ctx["M"]
n_obs = ctx["n_obs"]
n_rev = 1
n_cov = n_rev + 3
names = ["rev", "s", "sc", "c"]

model = ce.Model()
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov,
                   "n_simulations": N_SIMULATIONS},
    "subproblem": {"gurobi_params": {"TimeLimit": 5}},
}
model.load_config(cfg)
input_data = build_input_data_sp(ctx, R=R)
model.data.load_and_distribute_input_data(input_data)

# ── Evaluate ──────────────────────────────────────────────────────────
for theta in thetas:
    if model.comm_manager.is_root():
        t_str = np.array2string(theta, precision=2, separator=", ")
        print(f"\n  theta = {t_str}")
        gnorm_label = "|g|"
        header = f"  {'seed':>6}  {'f':>12}  " + "  ".join(f"{n:>10}" for n in names) + f"  {gnorm_label:>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

    for seed in error_seeds:
        cov_oracle, err_oracle = build_oracles(model, seed=seed,
                                               sigma_eps=SIGMA_EPS,
                                               sigma_nu_1=SIGMA_NU_1,
                                               sigma_nu_2=SIGMA_NU_2)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_oracle)
        model.features.set_error_oracle(err_oracle)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)
        if model.comm_manager.is_root():
            g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
            print(f"  {seed:>6}  {f_val:12.6f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
