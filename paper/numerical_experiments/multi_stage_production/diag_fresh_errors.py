#!/bin/env python
"""Diagnostic: N=100 S=1 single-rank, obj(theta_true) vs obj(theta_hat)."""

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

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)
from run_experiment import pad_x_Q, draw_simulation_errors

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
        'theta_bounds': {
            'lb': -20, 'ub': 20,
            'lbs': {0: -5, 1: -5, 2: -5, 3: -5},
            'ubs': {0: 5, 1: 5, 2: 5, 3: 5},
        },
    },
}
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)

# Draw FRESH simulation errors
ld = model.data.local_data
phi1, phi2, nu = draw_simulation_errors(
    model, firms, ng_max, P_max, nm_max, L1, L2, N,
    CFG['sigma'], err_seed=seed + 1000)
ld.errors['phi1'] = phi1
ld.errors['phi2'] = phi2
ld.errors['nu'] = nu

print(f"  Errors shapes: phi1={phi1.shape}, phi2={phi2.shape}, nu={nu.shape}")
print(f"  n_local_agent={model.comm_manager.num_local_agent}, n_obs={nf}")

# Set up oracles and solver
cov_oracle, err_oracle = build_oracles(
    model, geo, firms, ng_max, P_max, nm_max)
model.features.set_covariates_oracle(cov_oracle)
model.features.set_error_oracle(err_oracle)

model.subproblems.load_solver(MultiStageSolver)
model.subproblems.initialize_solver()

solver = model.subproblems.subproblem_solver
row_gen = model.point_estimation.n_slack

theta_true_vec = pack_theta(theta_true)
print(f"\n  theta_true = {theta_true_vec}")

# Run DC with 5 iters
dc = DCSolver(row_gen, solver)
result = dc.solve(theta_true_vec, max_dc_iters=10, tol=1e-4, verbose=True)

if result is not None:
    print(f"\n{'='*60}")
    print(f"  PARAMETER RECOVERY")
    print(f"{'='*60}")
    print(f"  {'Parameter':<15} {'True':>10} {'Estimated':>10} {'Error%':>10}")
    print(f"  {'-'*45}")
    for j, name in enumerate(THETA_NAMES):
        true_val = theta_true[name]
        est_val = result.theta_hat[j]
        err_pct = 100 * (est_val - true_val) / abs(true_val) if true_val != 0 else float('nan')
        print(f"  {name:<15} {true_val:>10.4f} {est_val:>10.4f} {err_pct:>9.1f}%")
    print(f"\n  DC converged: {result.converged}, iters: {result.num_iterations}, time: {result.total_time:.1f}s")
