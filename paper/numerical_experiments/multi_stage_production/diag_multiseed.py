#!/bin/env python
"""Multi-seed warm-start test."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import run
from solver import pack_theta, THETA_NAMES

BASE = Path(__file__).resolve().parent
with open(BASE / 'config.yaml') as f:
    CFG = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--n-firms', type=int, default=100)
parser.add_argument('--max-dc-iters', type=int, default=10)
args = parser.parse_args()

dgp = dict(
    n_firms=args.n_firms, n_markets=6, n_continents=3,
    l1_per_continent=[2, 2, 2],
    l2_per_continent=[2, 2, 2],
    n_groups_cells=2, n_platforms=4,
    models_range=[3, 8],
)

theta_true = CFG['theta_true']
theta_true_vec = pack_theta(theta_true)
n_params = len(THETA_NAMES)

seeds = [42, 43, 44, 45, 46]
estimates = np.zeros((len(seeds), n_params))
times = np.zeros(len(seeds))

for s_idx, seed in enumerate(seeds):
    print(f"\n{'#'*60}")
    print(f"  SEED {seed} (N={args.n_firms}, S=1, init=theta_true, max_dc={args.max_dc_iters})")
    print(f"{'#'*60}")
    result = run(dgp, seed=seed, n_simulations=1, max_dc_iters=args.max_dc_iters,
                 verbose_dc=False)
    if result is not None:
        estimates[s_idx] = result.theta_hat
        times[s_idx] = result.total_time

print(f"\n{'='*80}")
print(f"  MULTI-SEED SUMMARY: N={args.n_firms}, S=1, init=theta_true, {len(seeds)} seeds")
print(f"  bounds=[-100, 100], sigma_phi={CFG['sigma']['phi_1']}, sigma_nu={CFG['sigma']['nu']}")
print(f"{'='*80}")
print(f"  {'Parameter':<15} {'True':>8} {'Mean':>8} {'Std':>8} {'MeanErr%':>10}")
print(f"  {'-'*49}")
for j in range(n_params):
    t = theta_true_vec[j]
    m = estimates[:, j].mean()
    s = estimates[:, j].std()
    err = 100 * (m - t) / abs(t) if t != 0 else float('nan')
    print(f"  {THETA_NAMES[j]:<15} {t:>8.4f} {m:>8.4f} {s:>8.4f} {err:>9.1f}%")

print(f"\n  Per-seed estimates:")
print(f"  {'Parameter':<15}", end="")
for seed in seeds:
    print(f" {'s='+str(seed):>10}", end="")
print()
print(f"  {'-'*65}")
for j in range(n_params):
    print(f"  {THETA_NAMES[j]:<15}", end="")
    for s_idx in range(len(seeds)):
        print(f" {estimates[s_idx, j]:>10.4f}", end="")
    print()

# Report params near ±100
print(f"\n  Parameters within 1% of bounds (|val| > 99):")
any_bound = False
for s_idx, seed in enumerate(seeds):
    for j in range(n_params):
        if abs(estimates[s_idx, j]) > 99:
            print(f"    seed={seed}  {THETA_NAMES[j]:<15} = {estimates[s_idx, j]:.2f}")
            any_bound = True
if not any_bound:
    print(f"    (none)")

print(f"\n  Avg time per seed: {times.mean():.1f}s")
