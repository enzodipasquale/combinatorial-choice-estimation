#!/usr/bin/env python3
"""Run the 4 off-diagonal misspecification cells at one (N, J, S) setting.

Each cell pairs a DGP with a misspecified estimator:
  DGP=probit, est=logit    (logit MLE + combest gumbel oracle on probit data)
  DGP=logit,  est=probit   (probit MLE + combest normal oracle on logit data)

Output: results/misspec/raw/dgp_{dgp}_est_{est}.npz
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

MODELS = ["probit", "logit", "probit_corr"]
OFF_DIAGONAL_CELLS = [(d, e) for d in MODELS for e in MODELS if d != e]


def run_one(dgp, est, N, J, K, beta, n_reps, config, results_dir, solver):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    exp = config["experiment"]
    n_simulations = resolve_n_simulations(exp.get("n_simulations", 1), N)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DGP={dgp}, EST={est}, J={J}, N={N}, S={n_simulations}, "
              f"{n_reps} reps")
        print(f"{'='*60}", flush=True)

    betas_mle, betas_cb = [], []
    times_mle, times_cb = [], []
    n_failed = 0
    n_mle_not_converged = 0

    for rep in range(n_reps):
        try:
            res = run_replication(N, J, K, beta, replication=rep,
                                  config=config, dgp=dgp, est=est,
                                  solver=solver)
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

        if (rep + 1) % 20 == 0:
            err_mle = np.linalg.norm(b_mle - beta)
            err_cb = np.linalg.norm(b_cb - beta)
            print(f"  rep {rep}: mle_err={err_mle:.4f}  cb_err={err_cb:.4f}",
                  flush=True)

    if rank == 0:
        output_path = results_dir / f"dgp_{dgp}_est_{est}.npz"
        betas_mle = np.array(betas_mle)
        betas_cb = np.array(betas_cb)
        np.savez(output_path,
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
                 dgp=dgp, est=est,
                 solver=solver)
        R = len(betas_mle)
        bias_mle = np.linalg.norm(np.mean(betas_mle - beta, axis=0))
        bias_cb = np.linalg.norm(np.mean(betas_cb - beta, axis=0))
        rmse_mle = np.sqrt(np.mean((betas_mle - beta)**2))
        rmse_cb = np.sqrt(np.mean((betas_cb - beta)**2))
        print(f"\n  Saved {output_path.name}: R={R}  "
              f"||bias_MLE||={bias_mle:.3f}  ||bias_CB||={bias_cb:.3f}  "
              f"RMSE_MLE={rmse_mle:.3f}  RMSE_CB={rmse_cb:.3f}",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--J", type=int, default=10)
    parser.add_argument("--s-mode", choices=["one", "sqrt_n", "match_n"],
                        default="sqrt_n")
    parser.add_argument("--solver", choices=["one_slack", "n_slack"],
                        default="one_slack")
    args = parser.parse_args()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    s_mode_to_cfg = {"one": 1, "sqrt_n": "match_sqrt_N", "match_n": "match_N"}
    config["experiment"]["n_simulations"] = s_mode_to_cfg[args.s_mode]

    exp = config["experiment"]
    beta = np.array(exp["beta_star"])
    K = exp["K"]
    n_reps = exp["n_replications"]

    results_dir = SCRIPT_DIR / "results" / "misspec" / "raw"
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    for dgp, est in OFF_DIAGONAL_CELLS:
        run_one(dgp, est, args.N, args.J, K, beta, n_reps, config,
                results_dir, args.solver)


if __name__ == "__main__":
    main()
