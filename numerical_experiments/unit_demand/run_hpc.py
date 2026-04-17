#!/usr/bin/env python3
"""HPC entry point for unit-demand efficiency benchmarking.

Usage:
  mpiexec ... python run_hpc.py --model probit --s-mode match_n --N 200
  mpiexec ... python run_hpc.py --model logit  --s-mode sqrt_n  --N 500
  mpiexec ... python run_hpc.py --model probit --s-mode one     # all cells

Output: results/{model}/raw/{s_mode}/{model}_J{J}_N{N}.npz
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from numerical_experiments.unit_demand.run_experiment import (
    run_replication, resolve_n_simulations)

SCRIPT_DIR = Path(__file__).parent

S_MODE_TO_CONFIG = {
    "one": 1,
    "sqrt_n": "match_sqrt_N",
    "match_n": "match_N",
}


def run_cell(J, N, K, beta, n_reps, config, results_dir, model):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    exp = config["experiment"]
    n_simulations = resolve_n_simulations(exp.get("n_simulations", 1), N)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"{model}: J={J}, N={N}, S={n_simulations}, {n_reps} reps")
        print(f"{'='*60}", flush=True)

    betas_mle, betas_cb = [], []
    times_mle, times_cb = [], []
    n_failed = 0
    n_mle_not_converged = 0

    for rep in range(n_reps):
        try:
            res = run_replication(N, J, K, beta, replication=rep,
                                  config=config, model=model)
        except Exception as e:
            if rank == 0:
                print(f"  rep {rep}: FAILED ({e})", flush=True)
            n_failed += 1
            continue

        if rank != 0:
            continue

        b_mle = np.array(res["beta_mle"])
        b_cb = np.array(res["beta_combest"])

        if np.any(np.isnan(b_mle)) or np.any(np.abs(b_mle) > 50):
            print(f"  rep {rep}: MLE diverged "
                  f"(max|beta|={np.nanmax(np.abs(b_mle)):.1f})", flush=True)
            n_failed += 1
            continue

        if not res.get("mle_converged", True):
            n_mle_not_converged += 1

        betas_mle.append(b_mle)
        betas_cb.append(b_cb)
        times_mle.append(res["runtime_mle"])
        times_cb.append(res["runtime_combest"])

        if (rep + 1) % 10 == 0:
            err_mle = np.linalg.norm(b_mle - beta)
            err_cb = np.linalg.norm(b_cb - beta)
            print(f"  rep {rep}: mle_err={err_mle:.4f}  "
                  f"cb_err={err_cb:.4f}  t_mle={res['runtime_mle']:.1f}s  "
                  f"t_cb={res['runtime_combest']:.1f}s",
                  flush=True)

    if rank == 0:
        output_path = results_dir / f"{model}_J{J}_N{N}.npz"
        betas_mle = np.array(betas_mle)
        betas_cb = np.array(betas_cb)
        save_kwargs = dict(
            beta_mle=betas_mle,
            beta_cb=betas_cb,
            time_mle=np.array(times_mle),
            time_cb=np.array(times_cb),
            beta_star=beta,
            J=J, N=N, K=K,
            n_simulations=n_simulations,
            sigma=exp["sigma"],
            rho=exp["covariate_correlation"],
            n_failed=n_failed,
            n_mle_not_converged=n_mle_not_converged,
            model=model,
        )
        if model == "probit":
            save_kwargs["ghk_draws"] = exp.get("ghk_draws", 200)
        np.savez(output_path, **save_kwargs)

        R_valid = len(betas_mle)
        var_mle = np.var(betas_mle, axis=0, ddof=1).mean() if R_valid > 1 else 0
        var_cb = np.var(betas_cb, axis=0, ddof=1).mean() if R_valid > 1 else 0
        are = var_cb / var_mle if var_mle > 0 else np.inf
        print(f"\n  Saved {output_path.name}: R={R_valid}, failed={n_failed}, "
              f"mle_not_converged={n_mle_not_converged}, ARE(var)={are:.2f}",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["probit", "logit"],
                        required=True)
    parser.add_argument("--s-mode", choices=["one", "sqrt_n", "match_n"],
                        default="match_n")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--cell-index", type=int, default=None)
    args = parser.parse_args()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    config["experiment"]["n_simulations"] = S_MODE_TO_CONFIG[args.s_mode]

    exp = config["experiment"]
    beta = np.array(exp["beta_star"])
    K = exp["K"]
    n_reps = exp["n_replications"]
    J_values = config["grid"]["J"]
    N_values = config["grid"]["N"]
    cells = [(J, N) for J in J_values for N in N_values]

    results_dir = (SCRIPT_DIR / "results" / args.model / "raw" / args.s_mode)
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    if args.N is not None:
        for J in J_values:
            run_cell(J, args.N, K, beta, n_reps, config, results_dir,
                     args.model)
    elif args.cell_index is not None:
        J, N = cells[args.cell_index]
        run_cell(J, N, K, beta, n_reps, config, results_dir, args.model)
    else:
        for J, N in cells:
            run_cell(J, N, K, beta, n_reps, config, results_dir, args.model)


if __name__ == "__main__":
    main()
