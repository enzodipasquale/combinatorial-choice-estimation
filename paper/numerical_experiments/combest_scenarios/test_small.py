#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combest as ce
from paper.numerical_experiments.combest_scenarios.generate_data import generate_data
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"[Rank {rank}] Testing gross substitutes...")
if rank == 0:
    input_data, theta_star = generate_data('gross_substitutes', N=10, M=10, lambda_val=0.1, seed=42)
    print(f"[Rank {rank}] theta_star shape: {theta_star.shape}")
    print(f"[Rank {rank}] Lambda: {theta_star[0]:.4f}, Delta range: [{theta_star[1:].min():.2f}, {theta_star[1:].max():.2f}]")

    id_d, item_d = input_data["id_data"], input_data["item_data"]
    n_covariates = sum(
        d[k].shape[-1] for d in (id_d, item_d)
        for k in ("modular", "quadratic") if k in d
    )
    subproblem_name = "Greedy"
else:
    input_data = theta_star = n_covariates = subproblem_name = None

n_covariates = comm.bcast(n_covariates, root=0)
subproblem_name = comm.bcast(subproblem_name, root=0)
theta_star = comm.bcast(theta_star, root=0)

model = ce.Model()
model.load_config({
    "dimensions": {"n_obs": 10, "n_items": 10, "n_covariates": n_covariates, "n_simulations": 1},
    "subproblem": {"name": subproblem_name},
    "row_generation": {"max_iters": 50, "tolerance": 0.01, "theta_bounds": {"lb": -1000, "ub": 1000}},
})

model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()
model.features.build_local_modular_error_oracle(seed=1)
model.subproblems.load_solver()

model.subproblems.generate_obs_bundles(theta_star)

if rank == 0:
    obs = model.data.local_obs_bundles if hasattr(model.data, 'local_obs_bundles') else None
    if obs is not None:
        sizes = obs.sum(axis=1)
        print(f"[Rank {rank}] Bundle sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")

print(f"[Rank {rank}] Running point estimation...")
result = model.row_generation.solve(verbose=True)

if rank == 0:
    print(f"[Rank {rank}] Theta_hat: {result.theta_hat[:3]}")
    print(f"[Rank {rank}] Theta_star: {theta_star[:3]}")
    print(f"[Rank {rank}] Error (first 3): {result.theta_hat[:3] - theta_star[:3]}")

print(f"[Rank {rank}] Done.")
