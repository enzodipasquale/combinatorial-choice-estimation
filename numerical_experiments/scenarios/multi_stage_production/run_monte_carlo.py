#!/bin/env python
"""Monte Carlo driver: launches K independent mpirun jobs and aggregates.

For each seed:
  - Spawns a fresh `mpirun -n {ranks} python run_experiment.py --seed {seed}`
    subprocess so Gurobi state, MPI state, and combest model state are
    isolated between replications.
  - Parses the final "Results (N=..., S=..., seed=...)" block from stdout
    to extract theta_hat per parameter.

Collects all theta_hats and reports mean bias, std across seeds, RMSE.

Usage:
    python run_monte_carlo.py [--K 15] [--seed-base 1000] [--n-firms 18]
                              [--n-simulations 1] [--ranks 8]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent
PARAM_NAMES = ['delta_1', 'delta_2', 'rho_xi_1', 'rho_xi_2',
               'rho_HQ_1', 'rho_HQ_2', 'FE_1_r1', 'FE_1_r2',
               'FE_2_r1', 'FE_2_r2', 'rho_d_1', 'rho_d_2']


def parse_result_block(stdout: str):
    """Extract (theta_hat, converged) from the final 'Results' block."""
    # Find the Results header line
    m = re.search(r"Results \(N=(\d+), S=(\d+), seed=(\d+)\)", stdout)
    if not m:
        return None
    n_obs, n_sims, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Parse param lines: "  delta_1             2.0000     1.8646      -6.8%"
    theta_hat = {}
    for name in PARAM_NAMES:
        pat = rf"{name}\s+([-0-9.]+)\s+([-0-9.]+)\s+"
        pm = re.search(pat, stdout)
        if pm is None:
            return None
        theta_hat[name] = float(pm.group(2))

    conv_m = re.search(r"Converged:\s+(\w+)\s+Iters:\s+(\d+)\s+Time:\s+([0-9.]+)s",
                       stdout)
    converged = (conv_m.group(1).lower() == 'true') if conv_m else False
    iters = int(conv_m.group(2)) if conv_m else -1
    runtime = float(conv_m.group(3)) if conv_m else -1.0

    return {
        'n_obs': n_obs, 'n_sims': n_sims, 'seed': seed,
        'theta_hat': [theta_hat[n] for n in PARAM_NAMES],
        'converged': converged, 'iters': iters, 'runtime': runtime,
    }


def run_one_seed(seed: int, n_firms: int, n_sims: int, ranks: int,
                 log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed{seed}.log"
    cmd = [
        'mpirun', '-n', str(ranks),
        '.bundle/bin/python', str(BASE / 'run_experiment.py'),
        '--seed', str(seed),
        '--n-firms', str(n_firms),
        '--n-simulations', str(n_sims),
    ]
    t0 = time.perf_counter()
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(BASE.parents[2]))
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        return {'seed': seed, 'converged': False, 'theta_hat': [np.nan]*12,
                'iters': -1, 'runtime': elapsed, 'error': f"exit {proc.returncode}"}

    stdout = log_path.read_text()
    parsed = parse_result_block(stdout)
    if parsed is None:
        return {'seed': seed, 'converged': False, 'theta_hat': [np.nan]*12,
                'iters': -1, 'runtime': elapsed, 'error': "could not parse output"}
    return parsed


def aggregate_and_report(results: list, theta_true_vec: np.ndarray, K: int):
    theta_arr = np.array([r['theta_hat'] for r in results])
    conv_mask = np.array([r['converged'] for r in results])
    n_conv = int(conv_mask.sum())
    if n_conv == 0:
        print("\n  ALL REPLICATIONS FAILED")
        return None

    theta_conv = theta_arr[conv_mask]
    mean_hat = theta_conv.mean(axis=0)
    std_hat = theta_conv.std(axis=0, ddof=1) if n_conv > 1 else np.zeros(12)
    bias = mean_hat - theta_true_vec
    rmse = np.sqrt(bias**2 + std_hat**2)

    print(f"\n  Converged: {n_conv}/{K}")
    print(f"  {'Param':<12} {'True':>8} {'Mean':>8} {'Bias':>9} "
          f"{'Bias%':>7} {'Std':>7} {'RMSE':>7}")
    print(f"  {'-'*65}")
    for j, name in enumerate(PARAM_NAMES):
        tv = theta_true_vec[j]
        bias_pct = 100 * bias[j] / abs(tv) if tv != 0 else float('nan')
        print(f"  {name:<12} {tv:>8.3f} {mean_hat[j]:>8.3f} "
              f"{bias[j]:>+9.3f} {bias_pct:>+6.1f}% "
              f"{std_hat[j]:>7.3f} {rmse[j]:>7.3f}")

    return {
        'mean_hat': mean_hat.tolist(),
        'std_hat': std_hat.tolist(),
        'bias': bias.tolist(),
        'rmse': rmse.tolist(),
        'n_converged': n_conv,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--n-firms', type=int, default=None)
    parser.add_argument('--n-simulations', type=int, default=1)
    parser.add_argument('--ranks', type=int, default=8)
    args = parser.parse_args()

    with open(BASE / 'config.yaml') as f:
        cfg = yaml.safe_load(f)

    n_firms = args.n_firms if args.n_firms is not None else cfg['dgp']['n_firms']
    theta_true = cfg['theta_true']
    theta_true_vec = np.array([theta_true[n] for n in PARAM_NAMES])

    tag = f"mc_K{args.K}_N{n_firms}_S{args.n_simulations}"
    results_dir = BASE / 'results' / 'monte_carlo'
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / f"{tag}_logs"
    out_path = results_dir / f"{tag}.json"

    print(f"Monte Carlo: K={args.K}, N={n_firms}, S={args.n_simulations}, "
          f"ranks={args.ranks}")
    print(f"  Logs: {log_dir}")
    print(f"  Output: {out_path}\n")

    results = []
    t_start = time.perf_counter()
    for k in range(args.K):
        seed = args.seed_base + k
        print(f"[{k+1}/{args.K}] seed={seed} running...", flush=True)
        r = run_one_seed(seed, n_firms, args.n_simulations, args.ranks, log_dir)
        results.append(r)
        status = "OK" if r['converged'] else f"FAIL ({r.get('error', '?')})"
        print(f"  -> {status}, iters={r.get('iters')}, runtime={r.get('runtime', 0):.0f}s",
              flush=True)

        # Incremental save
        partial = {
            'theta_true': theta_true,
            'results': results,
            'K_completed': k + 1, 'K_total': args.K,
            'n_firms': n_firms, 'n_simulations': args.n_simulations,
            'param_names': PARAM_NAMES,
            'config': cfg,
        }
        with open(out_path, 'w') as f:
            json.dump(partial, f, indent=2)

    total = time.perf_counter() - t_start
    print(f"\n{'='*65}\n  MC complete. K={args.K}, total={total/60:.1f} min\n{'='*65}")
    agg = aggregate_and_report(results, theta_true_vec, args.K)

    if agg is not None:
        final = {
            'theta_true': theta_true,
            'results': results,
            'K_completed': args.K, 'K_total': args.K,
            'n_firms': n_firms, 'n_simulations': args.n_simulations,
            'param_names': PARAM_NAMES,
            'config': cfg,
            **agg,
        }
        with open(out_path, 'w') as f:
            json.dump(final, f, indent=2)
        print(f"\n  Saved -> {out_path}")


if __name__ == '__main__':
    main()
