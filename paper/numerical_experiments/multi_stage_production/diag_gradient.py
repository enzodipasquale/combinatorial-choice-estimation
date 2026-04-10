#!/bin/env python
"""Gradient check: analytic vs numerical FE gradient for each firm."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_synthetic_data
from costs import compute_rev_factor, compute_facility_costs
from solver import pack_theta, unpack_theta, THETA_NAMES

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

seed = 42
dgp = dict(
    n_firms=5, n_markets=6, n_continents=3,
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
FE_NAMES = ['FE_1_As', 'FE_1_Eu', 'FE_2_As', 'FE_2_Eu']


def compute_v(theta_vec, firm, bun, geo, coefs):
    """v(b_obs; theta) = pi(theta) * x_obs - fc1(theta) * y1 - fc2(theta) * y2"""
    theta_d = unpack_theta(theta_vec)
    rf = compute_rev_factor(geo, theta_d, coefs).transpose(2, 0, 1)  # (N, L1, L2)
    pi = (firm['shares'][:, :, None, None]
          * geo['R_n'][None, :, None, None]
          * rf[None, :, :, :])
    fc1, fc2 = compute_facility_costs(firm, geo, theta_d)
    return (pi * bun['x']).sum() - (fc1 * bun['y1']).sum() - (fc2 * bun['y2']).sum()


def analytic_grad_fe(theta_vec, firm, bun, geo, coefs):
    """dv/d(FE_k) for 4 FE params via the chain rule through rev_factor."""
    theta_d = unpack_theta(theta_vec)
    rf = compute_rev_factor(geo, theta_d, coefs).transpose(2, 0, 1)
    pi = (firm['shares'][:, :, None, None]
          * geo['R_n'][None, :, None, None]
          * rf[None, :, :, :])
    x = bun['x']
    c = (coefs['eta'] - 1) / abs(coefs['beta_2_T'])
    bp = coefs['beta_2_phi']
    grad = np.zeros(4)
    for k, cc in enumerate([1, 2]):
        mask = geo['cont1'] == cc
        grad[k] = c * bp * (pi[:, :, mask, :] * x[:, :, mask, :]).sum()
    for k, cc in enumerate([1, 2]):
        mask = geo['cont2'] == cc
        grad[2 + k] = c * (pi[:, :, :, mask] * x[:, :, :, mask]).sum()
    return grad


def numerical_grad_fe(theta_vec, firm, bun, geo, coefs, h=1e-5):
    grad = np.zeros(4)
    for k in range(4):
        e = np.zeros(len(theta_vec))
        e[k] = h
        grad[k] = (compute_v(theta_vec + e, firm, bun, geo, coefs)
                    - compute_v(theta_vec - e, firm, bun, geo, coefs)) / (2 * h)
    return grad


print("=" * 70)
print("  GRADIENT CHECK: analytic vs numerical (h=1e-5)")
print("=" * 70)
print(f"  c = (eta-1)/|beta_2_T| = {(coefs['eta'] - 1) / abs(coefs['beta_2_T']):.6f}")
print(f"  beta_2_phi = {coefs['beta_2_phi']}")
print(f"  cont1 = {geo['cont1'].tolist()}")
print(f"  cont2 = {geo['cont2'].tolist()}")

for f_idx in range(len(firms)):
    firm = firms[f_idx]
    bun = bundles[f_idx]
    v = compute_v(theta_true_vec, firm, bun, geo, coefs)
    ga = analytic_grad_fe(theta_true_vec, firm, bun, geo, coefs)
    gn = numerical_grad_fe(theta_true_vec, firm, bun, geo, coefs)

    print(f"\n  Firm {f_idx} (HQ={firm['hq_cont']}, nm={firm['n_models']}, "
          f"paths={(bun['x'] > 0.01).sum()}, v={v:.6f}):")
    print(f"    {'FE param':<12} {'Analytic':>14} {'Numerical':>14} {'Rel Err':>12}")
    print(f"    {'-'*52}")
    for k in range(4):
        a, n = ga[k], gn[k]
        rel_err = abs(a - n) / max(abs(n), 1e-15)
        tag = "OK" if rel_err < 1e-4 else "MISMATCH"
        print(f"    {FE_NAMES[k]:<12} {a:>14.8f} {n:>14.8f} {rel_err:>12.2e}  {tag}")

# Cross-check with h=1e-7 for firm 0
print(f"\n{'='*70}")
print(f"  CROSS-CHECK: firm 0, h=1e-7")
print(f"{'='*70}")
ga = analytic_grad_fe(theta_true_vec, firms[0], bundles[0], geo, coefs)
gn = numerical_grad_fe(theta_true_vec, firms[0], bundles[0], geo, coefs, h=1e-7)
print(f"    {'FE param':<12} {'Analytic':>14} {'Numerical':>14} {'Rel Err':>12}")
print(f"    {'-'*52}")
for k in range(4):
    a, n = ga[k], gn[k]
    rel_err = abs(a - n) / max(abs(n), 1e-15)
    print(f"    {FE_NAMES[k]:<12} {a:>14.8f} {n:>14.8f} {rel_err:>12.2e}")
