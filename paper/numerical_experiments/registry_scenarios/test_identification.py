#!/usr/bin/env python3
"""Check item identification across multiple seeds at N=100 (worst case)."""
import sys
import warnings
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
from mpi4py import MPI
import combest as ce
from paper.numerical_experiments.combest_scenarios.generate_data import generate_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

SPECS = {
    "gross_substitutes":  ("Greedy", {20: 0.005, 50: 0.002, 100: 0.002, 200: 0.002}),
    "supermodular":       ("QuadraticSupermodularMinCut", {20: 0.01, 50: 0.005, 100: 0.001, 200: 0.001}),
    "linear_knapsack":    ("LinearKnapsackGRB", None),
    "quadratic_knapsack": ("QuadraticKnapsackGRB", 0.05),
}

N = 100
alpha = 1.0
sigma = 1.0
SEEDS = [rep * 1000 + 42 for rep in range(5)]  # matches run_experiment.py seeding


def check(spec_name, M, lam, seed):
    subproblem, _ = SPECS[spec_name]
    input_data, theta_star = generate_data(spec_name, N, M, alpha=alpha, lambda_val=lam, seed=seed)
    id_d, item_d = input_data["id_data"], input_data["item_data"]
    n_cov = sum(d[k].shape[-1] for d in (id_d, item_d) for k in ("modular", "quadratic") if k in d)

    sub_cfg = {"name": subproblem, "gurobi_params": {"OutputFlag": 0, "Threads": 1}}
    if subproblem == "QuadraticKnapsackGRB":
        sub_cfg["gurobi_params"]["TimeLimit"] = 10.0

    rg_cfg = {"max_iters": 1, "tolerance": 0.01, "theta_bounds": {"lb": -100, "ub": 100}}
    if spec_name in ("supermodular", "quadratic_knapsack"):
        lbs = {str(idx): 0.0 for idx in input_data["meta"]["lambda_indices"]}
        rg_cfg["theta_bounds"]["lbs"] = lbs

    model = ce.Model()
    model.load_config({
        "dimensions": {"n_obs": N, "n_items": M, "n_covariates": n_cov, "n_simulations": 1},
        "subproblem": sub_cfg, "row_generation": rg_cfg,
    })
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    model.features.build_local_modular_error_oracle(seed=1, sigma=sigma)
    model.subproblems.load_solver()

    if spec_name == "gross_substitutes":
        def _fbi(lid, bun, il, th, bv, ld, me):
            M_ = len(bun)
            m = th[0] * ld.id_data["modular"][lid, :, 0] - th[1:M_+1] + me
            c = np.where(il)[0]
            j = c[np.argmax(m[c])]
            return j, bv + m[j] - 2 * th[M_+1] * int(bun.sum())
        model.subproblems.subproblem_solver.find_best_item = _fbi

    model.subproblems.generate_obs_bundles(theta_star)
    obs = model.data.local_obs_bundles
    freq = obs.sum(axis=0)
    return int((freq == 0).sum()), int((freq == N).sum())


Ms_by_spec = {
    "gross_substitutes": [50, 100, 200],
    "supermodular": [50, 100, 200],
    "linear_knapsack": [50, 100, 200],
    "quadratic_knapsack": [20],
}

if rank == 0:
    total_issues = 0
    for spec_name, (_, lam_cfg) in SPECS.items():
        for M in Ms_by_spec[spec_name]:
            lam = lam_cfg[M] if isinstance(lam_cfg, dict) else lam_cfg
            results = []
            for seed in SEEDS:
                n_never, n_always = check(spec_name, M, lam, seed)
                results.append((n_never, n_always))
            any_issue = any(nn > 0 or na > 0 for nn, na in results)
            total_issues += sum(nn + na for nn, na in results)
            status = "FAIL" if any_issue else "ok"
            detail = " ".join(f"({nn},{na})" for nn, na in results)
            print(f"[{status:>4}] {spec_name:>20} M={M:>3}  seeds: {detail}")
    print(f"\nTotal issues: {total_issues}")
