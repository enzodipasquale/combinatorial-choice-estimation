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

# Generate data (rank 0 only)
if rank == 0:
    input_data, theta_star = generate_data('gross_substitutes', N=200, M=20, lambda_val=0.005, seed=42)
    id_d, item_d = input_data["id_data"], input_data["item_data"]
    n_covariates = sum(
        d[k].shape[-1] for d in (id_d, item_d)
        for k in ("modular", "quadratic") if k in d
    )
else:
    input_data = theta_star = n_covariates = None

n_covariates = comm.bcast(n_covariates, root=0)
theta_star = comm.bcast(theta_star, root=0)

# Build model
model = ce.Model()
model.load_config({
    "dimensions": {"n_obs": 200, "n_items": 20, "n_covariates": n_covariates, "n_simulations": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 200, "tolerance": 0.01, "theta_bounds": {"lb": -100, "ub": 100}},
})
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()
model.features.build_local_modular_error_oracle(seed=1, sigma=1.0)
model.subproblems.load_solver()
model.subproblems.generate_obs_bundles(theta_star)
model.features.build_local_modular_error_oracle(seed=2, sigma=1.0)

# --- Row generation ---
if rank == 0:
    print("=" * 60)
    print("ROW GENERATION (n_slack)")
    print("=" * 60)
result_rg = model.point_estimation.n_slack.solve(verbose=True)
if rank == 0:
    print(f"  iters={result_rg.num_iterations}  obj={result_rg.final_objective:.6f}  time={result_rg.total_time:.3f}s")
    err_rg = np.linalg.norm(result_rg.theta_hat - theta_star)
    print(f"  ||theta_hat - theta_star|| = {err_rg:.6f}")

# --- Ellipsoid (same number of iters as row gen) ---
n_rg_iters = comm.bcast(result_rg.num_iterations if rank == 0 else None, root=0)
if rank == 0:
    print()
    print("=" * 60)
    print(f"ELLIPSOID ({n_rg_iters} iters, same as row gen)")
    print("=" * 60)
result_el = model.point_estimation.ellipsoid.solve(num_iters=n_rg_iters, verbose=True)
if rank == 0:
    print(f"  iters={result_el.num_iterations}  obj={result_el.final_objective:.6f}  time={result_el.total_time:.3f}s")
    err_el = np.linalg.norm(result_el.theta_hat - theta_star)
    print(f"  ||theta_hat - theta_star|| = {err_el:.6f}")

# --- Ellipsoid (auto iterations for convergence) ---
if rank == 0:
    print()
    print("=" * 60)
    print("ELLIPSOID (auto iters, precision=1e-2)")
    print("=" * 60)
result_el2 = model.point_estimation.ellipsoid.solve(precision=1e-2, verbose=True)
if rank == 0:
    print(f"  iters={result_el2.num_iterations}  obj={result_el2.final_objective:.6f}  time={result_el2.total_time:.3f}s")
    err_el2 = np.linalg.norm(result_el2.theta_hat - theta_star)
    print(f"  ||theta_hat - theta_star|| = {err_el2:.6f}")

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Row gen:    {result_rg.num_iterations:>5} iters  err={err_rg:.6f}  time={result_rg.total_time:.3f}s")
    print(f"  Ellipsoid:  {n_rg_iters:>5} iters  err={err_el:.6f}  time={result_el.total_time:.3f}s")
    print(f"  Ellipsoid:  {result_el2.num_iterations:>5} iters  err={err_el2:.6f}  time={result_el2.total_time:.3f}s")
