#!/usr/bin/env python3
"""Estimate asymptotic relative efficiency of combest vs probit MLE."""
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.unit_demand.run_experiment import run_replication


def run_grid(J_values, N_values, K, beta, n_reps, config):
    results = {}
    for J in J_values:
        for N in N_values:
            print(f"\n{'='*60}")
            print(f"J={J}, N={N}, {n_reps} replications")
            print(f"{'='*60}")

            betas_mle, betas_cb = [], []
            times_mle, times_cb = [], []
            n_failed = 0

            for r in range(n_reps):
                t0 = time.perf_counter()
                try:
                    res = run_replication('probit', N, J, K, beta,
                                          replication=r, config=config)
                except Exception as e:
                    print(f"  rep {r}: FAILED ({e})")
                    n_failed += 1
                    continue

                b_mle = np.array(res['beta_mle'])
                b_cb = np.array(res['beta_combest'])

                # skip if MLE produced garbage (e.g. huge values)
                if np.any(np.abs(b_mle) > 100):
                    print(f"  rep {r}: MLE diverged (max|beta|={np.abs(b_mle).max():.1f})")
                    n_failed += 1
                    continue

                betas_mle.append(b_mle)
                betas_cb.append(b_cb)
                times_mle.append(res['runtime_mle'])
                times_cb.append(res['runtime_combest'])

                elapsed = time.perf_counter() - t0
                if (r + 1) % 10 == 0 or r == 0:
                    print(f"  rep {r}: {elapsed:.1f}s  "
                          f"mle_err={np.linalg.norm(b_mle - beta):.4f}  "
                          f"cb_err={np.linalg.norm(b_cb - beta):.4f}")

            if len(betas_mle) < 5:
                print(f"  WARNING: only {len(betas_mle)} valid reps, skipping")
                continue

            betas_mle = np.array(betas_mle)
            betas_cb = np.array(betas_cb)
            R_valid = len(betas_mle)

            # MSE (per-component, then average)
            mse_mle = np.mean((betas_mle - beta) ** 2, axis=0)
            mse_cb = np.mean((betas_cb - beta) ** 2, axis=0)

            # Bias
            bias_mle = np.mean(betas_mle - beta, axis=0)
            bias_cb = np.mean(betas_cb - beta, axis=0)

            # Variance
            var_mle = np.var(betas_mle, axis=0)
            var_cb = np.var(betas_cb, axis=0)

            # Scalar summaries
            avg_mse_mle = mse_mle.mean()
            avg_mse_cb = mse_cb.mean()
            efficiency_ratio = avg_mse_cb / avg_mse_mle if avg_mse_mle > 0 else np.inf

            # N-scaled variance (should stabilize as N grows)
            n_var_mle = N * var_mle.mean()
            n_var_cb = N * var_cb.mean()

            results[(J, N)] = {
                'R': R_valid, 'n_failed': n_failed,
                'mse_mle': avg_mse_mle, 'mse_cb': avg_mse_cb,
                'bias_mle': bias_mle, 'bias_cb': bias_cb,
                'var_mle': var_mle.mean(), 'var_cb': var_cb.mean(),
                'n_var_mle': n_var_mle, 'n_var_cb': n_var_cb,
                'efficiency_ratio': efficiency_ratio,
                'time_mle': np.mean(times_mle), 'time_cb': np.mean(times_cb),
            }

            print(f"\n  Results (R={R_valid}, failed={n_failed}):")
            print(f"    MSE:  MLE={avg_mse_mle:.6f}  CB={avg_mse_cb:.6f}  ratio={efficiency_ratio:.2f}")
            print(f"    Bias: MLE={np.linalg.norm(bias_mle):.6f}  CB={np.linalg.norm(bias_cb):.6f}")
            print(f"    N*Var: MLE={n_var_mle:.4f}  CB={n_var_cb:.4f}")
            print(f"    Time: MLE={np.mean(times_mle):.2f}s  CB={np.mean(times_cb):.3f}s")

    return results


def print_summary_table(results, J_values, N_values):
    print(f"\n{'='*80}")
    print("EFFICIENCY SUMMARY: MSE_combest / MSE_mle")
    print(f"{'='*80}")
    header = f"{'J':>4} |" + "".join(f"  N={N:>5}  |" for N in N_values)
    print(header)
    print("-" * len(header))
    for J in J_values:
        row = f"{J:>4} |"
        for N in N_values:
            r = results.get((J, N))
            if r:
                row += f"  {r['efficiency_ratio']:>6.2f}  |"
            else:
                row += f"  {'--':>6}  |"
        print(row)

    print(f"\n{'='*80}")
    print("N * Variance (should stabilize as N grows)")
    print(f"{'='*80}")
    header = f"{'J':>4} |" + "".join(f"  N={N:>5}  |" for N in N_values)
    print(header)
    print("-" * len(header))
    for J in J_values:
        row_mle = f"{'':>4} |"
        row_cb = f"{J:>4} |"
        for N in N_values:
            r = results.get((J, N))
            if r:
                row_mle += f"  {r['n_var_mle']:>6.2f}  |"
                row_cb += f"  {r['n_var_cb']:>6.2f}  |"
            else:
                row_mle += f"  {'--':>6}  |"
                row_cb += f"  {'--':>6}  |"
        print(f"  MLE{row_mle}")
        print(f"  CB {row_cb}")
        print()


if __name__ == "__main__":
    J_values = [2, 5, 10]
    N_values = [500, 1000, 2000]
    K = 3
    beta = np.array([1.0, -0.5, 0.5])
    n_reps = 30

    config = {
        'experiment': {'n_simulations': 1, 'sigma': 1.0, 'ghk_draws': 200},
        'row_generation': {'max_iters': 200, 'tolerance': 0.01,
                           'theta_bounds': {'lb': -10, 'ub': 10}},
        'subproblem': {'gurobi_params': {'OutputFlag': 0, 'Threads': 1}},
    }

    results = run_grid(J_values, N_values, K, beta, n_reps, config)
    print_summary_table(results, J_values, N_values)
