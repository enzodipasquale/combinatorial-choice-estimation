#!/bin/env python
import sys
from pathlib import Path
import yaml
import numpy as np
import combest as ce
from solver import TwoStageSolver
from oracles import build_oracles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
from prepare_data import main as load_data, build_input_data

with open(Path(__file__).resolve().parent / "test_config.yaml") as f:
    CFG = yaml.safe_load(f)["test_real_data"]

COUNTRY = CFG["country"]
KEEP_TOP = CFG["keep_top"]
END_BUFFER = CFG["end_buffer"]
N_SAMPLE = CFG["n_sample"]
N_SIMULATIONS = CFG["n_simulations"]
BETA = CFG["beta"]
R = CFG["R"]
SIGMA_1 = CFG["sigma_1"]
SIGMA_2 = CFG["sigma_2"]
ERROR_SEEDS = CFG["error_seeds"]

theta_0 = np.array([1.47413693, -2.89827683, -0.01765633, 0.0, 0.07047045])
thetas = [theta_0, np.zeros(5)]

ctx = load_data(COUNTRY, KEEP_TOP, end_buffer=END_BUFFER,
                n_sample=N_SAMPLE)
M = ctx["M"]
n_obs = ctx["n_obs"]
n_rev = 1
n_cov = n_rev + 4
names = ["rev", "entry_c", "entry_dist", "syn", "syn_d"]

model = ce.Model()
cfg = {
    "dimensions": {"n_obs": n_obs, "n_items": M, "n_covariates": n_cov,
                   "n_simulations": N_SIMULATIONS},
    "subproblem": {"gurobi_params": {"TimeLimit": 5}},
}
model.load_config(cfg)
if model.comm_manager.is_root():
    input_data = build_input_data(ctx, R=R)
else:
    input_data = None
model.data.load_and_distribute_input_data(input_data)

for theta in thetas:
    if model.comm_manager.is_root():
        t_str = np.array2string(theta, precision=2, separator=", ")
        print(f"\n  theta = {t_str}")
        gnorm_label = "|g|"
        header = f"  {'seed':>6}  {'f':>12}  " + "  ".join(f"{n:>10}" for n in names) + f"  {gnorm_label:>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

    for seed in ERROR_SEEDS:
        cov_oracle, err_oracle = build_oracles(model, beta=BETA, seed=seed,
                                               sigma_1=SIGMA_1,
                                               sigma_2=SIGMA_2)
        model.subproblems.load_solver(TwoStageSolver)
        model.subproblems.initialize_solver()
        model.features.set_covariates_oracle(cov_oracle)
        model.features.set_error_oracle(err_oracle)

        f_val, g_val = model.point_estimation.compute_nonlinear_obj_and_grad_at_root(theta)
        if model.comm_manager.is_root():
            g_str = "  ".join(f"{g_val[j]:+10.4f}" for j in range(n_cov))
            print(f"  {seed:>6}  {f_val:12.6f}  {g_str}  {np.linalg.norm(g_val):10.4f}")
