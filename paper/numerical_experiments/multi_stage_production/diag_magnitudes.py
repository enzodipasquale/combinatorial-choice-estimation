#!/bin/env python
"""Diagnostic: inspect magnitudes of structural signal vs noise."""

import sys
from pathlib import Path
import yaml
import numpy as np
import combest as ce

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from costs import compute_rev_factor, compute_facility_costs
from solver import (MultiStageSolver, flatten_bundle, pack_theta, unpack_theta,
                    THETA_NAMES)
from oracles import build_oracles, N_PARAMS
from run_experiment import pad_x_Q, draw_simulation_errors

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

seed = 42
dgp = dict(
    n_firms=10, n_markets=6, n_continents=3,
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

coefs = CFG['sourcing_coefs']
theta_true_vec = pack_theta(theta_true)
nf = len(firms)
L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']

# ---- PART 1: structural utility decomposition ----
print("=" * 70)
print("  PART 1: STRUCTURAL UTILITY DECOMPOSITION AT theta_true")
print("=" * 70)

rf = compute_rev_factor(geo, theta_true, coefs).transpose(2, 0, 1)  # (N, L1, L2)

v_pi = np.zeros(nf)
v_fc1 = np.zeros(nf)
v_fc2 = np.zeros(nf)
e_nu = np.zeros(nf)
e_phi1 = np.zeros(nf)
e_phi2 = np.zeros(nf)

for i, (firm, bun) in enumerate(zip(firms, bundles)):
    pi = (firm['shares'][:, :, None, None]
          * geo['R_n'][None, :, None, None]
          * rf[None, :, :, :])
    fc1, fc2 = compute_facility_costs(firm, geo, theta_true)

    v_pi[i] = (pi * bun['x']).sum()
    v_fc1[i] = (fc1 * bun['y1']).sum()
    v_fc2[i] = (fc2 * bun['y2']).sum()
    e_nu[i] = (bun['nu'] * bun['x']).sum()
    e_phi1[i] = (bun['phi1'] * bun['y1']).sum()
    e_phi2[i] = (bun['phi2'] * bun['y2']).sum()

v_total = v_pi - v_fc1 - v_fc2
e_total = e_nu - e_phi1 - e_phi2

print(f"\n  {'Firm':<6} {'pi*x':>10} {'-fc1*y1':>10} {'-fc2*y2':>10} {'v_det':>10}"
      f" {'nu*x':>10} {'-phi1*y1':>10} {'-phi2*y2':>10} {'eps':>10}")
print(f"  {'-'*86}")
for i in range(nf):
    print(f"  {i:<6} {v_pi[i]:>10.3f} {-v_fc1[i]:>10.3f} {-v_fc2[i]:>10.3f} {v_total[i]:>10.3f}"
          f" {e_nu[i]:>10.3f} {-e_phi1[i]:>10.3f} {-e_phi2[i]:>10.3f} {e_total[i]:>10.3f}")

print(f"\n  v_det:  mean={v_total.mean():.3f}  std={v_total.std():.3f}  "
      f"range={v_total.max()-v_total.min():.3f}")
print(f"  eps:    mean={e_total.mean():.3f}  std={e_total.std():.3f}  "
      f"range={e_total.max()-e_total.min():.3f}")
print(f"\n  Signal/noise ratio (std(v_det)/std(eps)): {v_total.std()/e_total.std():.3f}")

# ---- PART 2: feature magnitudes from combest oracles ----
print(f"\n{'='*70}")
print(f"  PART 2: FEATURE MAGNITUDES (covariates & error oracles)")
print(f"{'='*70}")

ng_max = max(len(f['ln_xi_1']) for f in firms)
P_max = max(f['n_platforms'] for f in firms)
nm_max = max(f['n_models'] for f in firms)
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
        'theta_bounds': {'lb': -20, 'ub': 20,
                         'lbs': {0: -5, 1: -5, 2: -5, 3: -5},
                         'ubs': {0: 5, 1: 5, 2: 5, 3: 5}},
    },
}
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)

# Fresh simulation errors
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

# Set Q-linearization at theta_true so oracles can evaluate
solver = model.subproblems.subproblem_solver
solver.set_q_linearization(theta_true_vec)

# Also solve at theta_true so lin_V is populated (needed if oracle falls through)
solver.solve(theta_true_vec)

# Use the actual obs_bundles object from combest (identity check in oracle)
local_obs = model.data.local_obs_bundles
ids = np.arange(nf)
cov_obs = cov_oracle(local_obs, ids, ld)
err_obs = err_oracle(local_obs, ids, ld)

print(f"\n  cov_obs shape: {cov_obs.shape}")
print(f"\n  {'Column':<15} {'Name':<15} {'Std':>10} {'Max|val|':>10} {'Mean':>10}")
print(f"  {'-'*60}")
for j in range(N_PARAMS):
    col = cov_obs[:, j]
    print(f"  {j:<15} {THETA_NAMES[j]:<15} {col.std():>10.4f} {np.abs(col).max():>10.4f} {col.mean():>10.4f}")

print(f"\n  error_obs: mean={err_obs.mean():.4f}  std={err_obs.std():.4f}  "
      f"max|val|={np.abs(err_obs).max():.4f}")

print(f"\n  FE columns (0-3) max|val|:")
for j in range(4):
    print(f"    {THETA_NAMES[j]:<12} max|cov|={np.abs(cov_obs[:,j]).max():.6f}  "
          f"  vs  std(err)={err_obs.std():.6f}  "
          f"  ratio={np.abs(cov_obs[:,j]).max()/err_obs.std():.4f}")

# ---- PART 3: sensitivity check ----
print(f"\n{'='*70}")
print(f"  PART 3: SENSITIVITY CHECK")
print(f"{'='*70}")


def v_all_firms(theta_vec):
    theta_d = unpack_theta(theta_vec)
    rf_loc = compute_rev_factor(geo, theta_d, coefs).transpose(2, 0, 1)
    vals = np.zeros(nf)
    for i, (firm, bun) in enumerate(zip(firms, bundles)):
        pi = (firm['shares'][:, :, None, None]
              * geo['R_n'][None, :, None, None]
              * rf_loc[None, :, :, :])
        fc1, fc2 = compute_facility_costs(firm, geo, theta_d)
        vals[i] = ((pi * bun['x']).sum()
                    - (fc1 * bun['y1']).sum()
                    - (fc2 * bun['y2']).sum())
    return vals


v_base = v_all_firms(theta_true_vec)

# Perturb FE_1_As: 0.5 -> 1.0
theta_p = theta_true_vec.copy()
theta_p[0] += 0.5
v_pert_fe1 = v_all_firms(theta_p)
dv_fe1 = v_pert_fe1 - v_base

print(f"\n  Perturb FE_1_As: {theta_true['FE_1_As']} -> {theta_true['FE_1_As']+0.5}")
print(f"    dv per firm: {dv_fe1}")
print(f"    mean|dv| = {np.abs(dv_fe1).mean():.6f}")
print(f"    std(eps)  = {e_total.std():.6f}")
print(f"    mean|dv| / std(eps) = {np.abs(dv_fe1).mean() / e_total.std():.4f}")

# Perturb delta_2_As: 0.03 -> 0.13
theta_p2 = theta_true_vec.copy()
theta_p2[8] += 0.1  # delta_2_As is index 8
v_pert_d2 = v_all_firms(theta_p2)
dv_d2 = v_pert_d2 - v_base

print(f"\n  Perturb delta_2_As: {theta_true['delta_2_As']} -> {theta_true['delta_2_As']+0.1}")
print(f"    dv per firm: {dv_d2}")
print(f"    mean|dv| = {np.abs(dv_d2).mean():.6f}")
print(f"    std(eps)  = {e_total.std():.6f}")
print(f"    mean|dv| / std(eps) = {np.abs(dv_d2).mean() / e_total.std():.4f}")

# Also perturb rho_xi_1 for comparison (well-recovered param)
theta_p3 = theta_true_vec.copy()
theta_p3[12] += 0.5  # rho_xi_1 is index 12
v_pert_rho = v_all_firms(theta_p3)
dv_rho = v_pert_rho - v_base

print(f"\n  Perturb rho_xi_1: {theta_true['rho_xi_1']} -> {theta_true['rho_xi_1']+0.5}")
print(f"    dv per firm: {dv_rho}")
print(f"    mean|dv| = {np.abs(dv_rho).mean():.6f}")
print(f"    std(eps)  = {e_total.std():.6f}")
print(f"    mean|dv| / std(eps) = {np.abs(dv_rho).mean() / e_total.std():.4f}")
