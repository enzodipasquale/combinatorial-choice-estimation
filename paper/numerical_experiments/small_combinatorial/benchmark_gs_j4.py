#!/usr/bin/env python3
"""Benchmark: GS, J=4, N=1000, S=1 vs S=sqrt(N), 100 replications."""
import gc
import sys
import json
import math
from pathlib import Path
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.small_combinatorial.run_experiment import run_replication

SCRIPT_DIR = Path(__file__).parent

N_SIMULATIONS_LIST = [1, int(math.isqrt(1000))]  # S=1, S=sqrt(1000)=31
N_REPS = 100
SPEC = "gross_substitutes"
J = 4
N = 1000


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    spec_cfg = config["specifications"][SPEC]
    alpha = spec_cfg.get("alpha", 1.0)
    lambda_val = spec_cfg["lambda"][J]

    results_dir = SCRIPT_DIR / "results" / "benchmark_gs_j4"
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    for S in N_SIMULATIONS_LIST:
        label = f"S{S}"
        if rank == 0:
            print(f"\n{'='*60}", flush=True)
            print(f"GS J={J} N={N} S={S}", flush=True)
            print(f"{'='*60}", flush=True)

        for rep in range(N_REPS):
            rep_file = results_dir / f"{label}_rep{rep}.json"
            rep_exists = rep_file.exists() if rank == 0 else False
            rep_exists = comm.bcast(rep_exists, root=0)
            if rep_exists:
                continue

            if rank == 0:
                print(f"  Rep {rep+1}/{N_REPS}...", end=" ", flush=True)

            result = run_replication(
                SPEC, N, J, alpha=alpha, lambda_val=lambda_val,
                replication=rep, config=config, n_simulations=S)

            if result is not None:
                result["n_simulations"] = S
                with open(rep_file, "w") as f:
                    json.dump(result, f, indent=2)
                print("✓", flush=True)

            gc.collect()

    # Print ARE summary on rank 0
    if rank == 0:
        import numpy as np
        print(f"\n{'='*60}")
        print("ARE Summary (MSE_MLE / MSE_combest)")
        print(f"{'='*60}")
        for S in N_SIMULATIONS_LIST:
            label = f"S{S}"
            files = sorted(results_dir.glob(f"{label}_rep*.json"))
            errs_mle, errs_cb = [], []
            for f in files:
                d = json.load(open(f))
                ts = np.array(d["theta_star"])
                tm = np.array(d["theta_mle"])
                tc = np.array(d["theta_combest"])
                ai = d.get("alpha_indices", [0])
                di = [i for i in range(len(ts)) if i not in ai]
                errs_mle.append(np.mean((tm[di] - ts[di])**2))
                errs_cb.append(np.mean((tc[di] - ts[di])**2))
            mse_m = np.mean(errs_mle)
            mse_c = np.mean(errs_cb)
            are = mse_m / mse_c
            print(f"  S={S:3d}: RMSE_mle={np.sqrt(mse_m):.4f}  RMSE_cb={np.sqrt(mse_c):.4f}  ARE={are:.3f}  ({len(files)} reps)")


if __name__ == "__main__":
    main()
