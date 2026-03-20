#!/usr/bin/env python3
"""
HPC entry point for probit efficiency benchmarking.
Runs all (J, N) cells × R replications, saves raw .npz per cell.
"""
import sys
import time
from pathlib import Path
import numpy as np
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.unit_demand.run_experiment import run_replication

SCRIPT_DIR = Path(__file__).parent


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    beta = np.array(exp["beta_star"])
    K = exp["K"]
    n_reps = exp["n_replications"]
    J_values = config["grid"]["J"]
    N_values = config["grid"]["N"]

    results_dir = SCRIPT_DIR / "results" / "raw"
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    for J in J_values:
        for N in N_values:
            output_path = results_dir / f"probit_J{J}_N{N}.npz"

            if rank == 0:
                print(f"\n{'='*60}")
                print(f"J={J}, N={N}, {n_reps} replications")
                print(f"{'='*60}", flush=True)

            betas_mle, betas_cb = [], []
            times_mle, times_cb = [], []
            n_failed = 0

            for rep in range(n_reps):
                try:
                    res = run_replication(N, J, K, beta, replication=rep,
                                          config=config)
                except Exception as e:
                    if rank == 0:
                        print(f"  rep {rep}: FAILED ({e})", flush=True)
                    n_failed += 1
                    continue

                if rank == 0:
                    b_mle = np.array(res["beta_mle"])
                    b_cb = np.array(res["beta_combest"])

                    # skip diverged MLE
                    if np.any(np.abs(b_mle) > 50):
                        print(f"  rep {rep}: MLE diverged", flush=True)
                        n_failed += 1
                        continue

                    betas_mle.append(b_mle)
                    betas_cb.append(b_cb)
                    times_mle.append(res["runtime_mle"])
                    times_cb.append(res["runtime_combest"])

                    if (rep + 1) % 10 == 0:
                        print(f"  rep {rep}: "
                              f"mle_err={np.linalg.norm(b_mle - beta):.4f}  "
                              f"cb_err={np.linalg.norm(b_cb - beta):.4f}  "
                              f"t_mle={res['runtime_mle']:.1f}s", flush=True)

            if rank == 0:
                np.savez(output_path,
                         beta_mle=np.array(betas_mle),
                         beta_cb=np.array(betas_cb),
                         time_mle=np.array(times_mle),
                         time_cb=np.array(times_cb),
                         beta_star=beta,
                         J=J, N=N, K=K,
                         sigma=exp["sigma"],
                         rho=exp["covariate_correlation"],
                         n_failed=n_failed)
                R_valid = len(betas_mle)
                mse_mle = np.mean((np.array(betas_mle) - beta)**2)
                mse_cb = np.mean((np.array(betas_cb) - beta)**2)
                print(f"\n  Saved {output_path.name}: "
                      f"R={R_valid}, failed={n_failed}, "
                      f"ARE={mse_cb/mse_mle:.2f}", flush=True)

    if rank == 0:
        print("\nDone. Run analyze_results.py locally to generate figures.")


if __name__ == "__main__":
    main()
