#!/usr/bin/env python3
"""Run all small combinatorial experiments: MLE vs combest."""
import gc
import sys
import json
from pathlib import Path
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.small_combinatorial.run_experiment import run_replication

SCRIPT_DIR = Path(__file__).parent


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    grid_J = config["grid"]["J"]
    grid_N = config["grid"]["N"]
    specs = ["gross_substitutes", "supermodular"]
    n_reps = config["experiment"]["n_replications"]

    results_dir = SCRIPT_DIR / "results" / "raw"
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    for spec in specs:
        spec_cfg = config["specifications"][spec]
        alpha = spec_cfg.get("alpha", 1.0)
        lambda_cfg = spec_cfg.get("lambda")

        for J in grid_J:
            resolve = lambda c, k: c.get(k, c.get(str(k))) if isinstance(c, dict) else c
            lambda_val = resolve(lambda_cfg, J) if isinstance(lambda_cfg, dict) else lambda_cfg

            for N in grid_N:
                # Skip if all reps exist
                existing = [results_dir / f"{spec}_J{J}_N{N}_rep{r}.json"
                            for r in range(n_reps)]
                n_done = sum(1 for f in existing if f.exists()) if rank == 0 else 0
                n_done = comm.bcast(n_done, root=0)
                if n_done >= n_reps:
                    if rank == 0:
                        print(f"Skipping {spec} J={J} N={N} ({n_done}/{n_reps} done)",
                              flush=True)
                    continue

                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f"{spec}, J={J}, N={N}, α={alpha}, λ={lambda_val}")
                    print(f"{'='*60}", flush=True)

                for rep in range(n_reps):
                    rep_file = results_dir / f"{spec}_J{J}_N{N}_rep{rep}.json"
                    rep_exists = rep_file.exists() if rank == 0 else False
                    rep_exists = comm.bcast(rep_exists, root=0)
                    if rep_exists:
                        continue

                    if rank == 0:
                        print(f"  Rep {rep+1}/{n_reps}...", end=" ", flush=True)

                    result = run_replication(
                        spec, N, J, alpha=alpha, lambda_val=lambda_val,
                        replication=rep, config=config)

                    if result is not None:
                        with open(rep_file, "w") as f:
                            json.dump(result, f, indent=2)
                        print("✓", flush=True)

                gc.collect()


if __name__ == "__main__":
    main()
