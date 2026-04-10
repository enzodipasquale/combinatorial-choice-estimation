#!/bin/env python
"""Diagnostic: f(theta_true) vs f(theta_hat) at N=100, seed=42."""

import sys
from pathlib import Path
import yaml
import numpy as np
import combest as ce

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from solver import MultiStageSolver, flatten_bundle, pack_theta, THETA_NAMES
from oracles import build_oracles, N_PARAMS
from dc import DCSolver
from run_experiment import pad_x_Q, draw_simulation_errors

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

seed = 42
dgp = dict(
    n_firms=100, n_markets=6, n_continents=3,
    l1_per_continent=[2, 2, 2],
    l2_per_continent=[2, 2, 2],
    n_groups_cells=2, n_platforms=4,
    models_range=[3, 8],
)

geo, firms, bundles, theta_true = generate_synthetic_data(
    seed=seed, dgp=dgp,
    sourcing_coefs=CFG['sourcing_coefs'],
    theta_true=CFG['theta_true'],
    sigma=CFG['sigma'],
)

nf = len(firms)
ng_max = max(len(f['ln_xi_1']) for f in firms)
P_max = max(f['n_platforms'] for f in firms)
nm_max = max(f['n_models'] for f in firms)
L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
n_items = ng_max * L1 + P_max * L2 + nm_max * N

obs_bundles = np.stack([
    flatten_bundle(bun, ng_max, P_max, nm_max, L1, L2, N)
    for bun in bundles
])
x_Q_global = pad_x_Q(bundles, firms, nm_max, N, L1, L2)

model = ce.Model()
input_data = {
    'id_data': {
        'obs_bundles': obs_bundles, 'firms': firms, 'x_Q': x_Q_global,
    },
    'item_data': {
        'geo': geo, 'sourcing_coefs': CFG['sourcing_coefs'],
        'ng_max': ng_max, 'P_max': P_max, 'nm_max': nm_max,
    },
}
cfg = {
    'dimensions': {
        'n_obs': nf, 'n_items': n_items,
        'n_covariates': N_PARAMS, 'n_simulations': 1,
    },
    'subproblem': {'gurobi_params': {'TimeLimit': 60}},
    'row_generation': {
        'max_iters': 200, 'tolerance': 1e-6,
        'theta_bounds': {'lb': -100, 'ub': 100},
    },
}
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)

ld = model.data.local_data
phi1, phi2, nu = draw_simulation_errors(
    model, firms, ng_max, P_max, nm_max, L1, L2, N,
    CFG['sigma'], err_seed=seed + 1000)
ld.errors['phi1'] = phi1
ld.errors['phi2'] = phi2
ld.errors['nu'] = nu

cov_oracle, err_oracle = build_oracles(
    model, geo, firms, ng_max, P_max, nm_max)
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

model.subproblems.load_solver(MultiStageSolver)
model.subproblems.initialize_solver()

solver = model.subproblems.subproblem_solver
row_gen = model.point_estimation.n_slack
local_obs = model.data.local_obs_bundles

theta_true_vec = pack_theta(theta_true)
ids = np.arange(nf)


def evaluate_f(theta_vec, label):
    """Evaluate f(theta) = sol_term - obs_term."""
    # Set linearizations and solve subproblems
    solver.set_q_linearization(theta_vec)
    solver_bundles = solver.solve(theta_vec)

    # Obs term: cov and err on observed bundles (uses lin_Q)
    cov_obs = cov_oracle(local_obs, ids, ld)
    err_obs = err_oracle(local_obs, ids, ld)
    obs_term = (cov_obs @ theta_vec + err_obs).sum()

    # Solver term: cov and err on solver bundles (uses lin_V)
    cov_v = cov_oracle(solver_bundles, ids, ld)
    err_v = err_oracle(solver_bundles, ids, ld)
    sol_term = (cov_v @ theta_vec + err_v).sum()

    f_val = sol_term - obs_term

    print(f"\n  {label}:")
    print(f"    theta = {theta_vec}")
    print(f"    obs_term = {obs_term:.6f}")
    print(f"    sol_term = {sol_term:.6f}")
    print(f"    f(theta) = {f_val:.6f}")

    return f_val


# First run DC to get theta_hat
print("=" * 60)
print("  Running DC to get theta_hat...")
print("=" * 60)

dc = DCSolver(row_gen, solver)
result = dc.solve(theta_true_vec, max_dc_iters=10, tol=1e-3, verbose=False)
theta_hat = result.theta_hat

print(f"\n  DC converged: {result.converged}, iters: {result.num_iterations}")
print(f"  theta_hat = {theta_hat}")

# Now evaluate f at both points
print(f"\n{'='*60}")
print(f"  OBJECTIVE EVALUATION")
print(f"{'='*60}")

f_true = evaluate_f(theta_true_vec, "f(theta_true)")
f_hat = evaluate_f(theta_hat, "f(theta_hat)")

print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
print(f"  f(theta_true) = {f_true:.6f}")
print(f"  f(theta_hat)  = {f_hat:.6f}")
print(f"  f(theta_hat) - f(theta_true) = {f_hat - f_true:.6f}")
print(f"  f(theta_hat) < f(theta_true)? {f_hat < f_true}")
