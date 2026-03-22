#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from itertools import product
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.combest_scenarios.run_experiment import run_replication
from paper.numerical_experiments.combest_scenarios.compute_statistics import load_replication_results, compute_statistics
from paper.numerical_experiments.combest_scenarios.aggregate_results import (
    load_all_statistics, generate_table_csv, generate_table_latex,
)

SCRIPT_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=None,
                        help="N values to run. If omitted, runs all from config.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with open(SCRIPT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    grid_M = config["grid"]["M"]
    grid_N = args.N if args.N else config["grid"]["N"]
    specs = list(config["specifications"].keys())
    n_reps = config["experiment"]["n_replications"]

    results_dir = SCRIPT_DIR / "results" / "raw"
    stats_dir = SCRIPT_DIR / "results" / "hpc"
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Run smaller M values first across all specs, then M=200 last.
    # This ensures partial results are usable if the job times out.
    max_M_first = max(grid_M)
    m_order = [m for m in grid_M if m < max_M_first] + [max_M_first]

    for M in m_order:
        for spec in specs:
            spec_cfg = config["specifications"][spec]
            max_M = spec_cfg.get("max_M", 200)
            if M > max_M:
                if rank == 0:
                    print(f"Skipping {spec} M={M} (max_M={max_M})")
                continue

            alpha_cfg = spec_cfg.get("alpha")
            lambda_cfg = spec_cfg.get("lambda")
            resolve = lambda c, M: c.get(M, c.get(str(M))) if isinstance(c, dict) else c
            alpha_val = resolve(alpha_cfg, M) if isinstance(alpha_cfg, dict) else alpha_cfg
            lambda_val = resolve(lambda_cfg, M) if isinstance(lambda_cfg, dict) else lambda_cfg

            for N in grid_N:
                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f"Running {spec}, N={N}, M={M}, \u03b1={alpha_val}, \u03bb={lambda_val}")
                    print(f"{'='*60}", flush=True)

                for rep in range(n_reps):
                    if rank == 0:
                        print(f"  Replication {rep+1}/{n_reps}...", end=" ", flush=True)

                    result = run_replication(
                        spec, N, M, alpha=alpha_val, lambda_val=lambda_val,
                        replication=rep, config=config,
                    )

                    if result is not None:
                        out = results_dir / f"{spec}_N{N}_M{M}_rep{rep}.json"
                        with open(out, "w") as f:
                            json.dump(result, f, indent=2)
                        print("\u2713", flush=True)

                if rank == 0:
                    results = load_replication_results(results_dir, spec, N, M)
                    stats = compute_statistics(results)
                    with open(stats_dir / f"stats_{spec}_N{N}_M{M}.json", "w") as f:
                        json.dump(stats, f, indent=2)
                    print(f"  Stats: {stats['n_replications']} reps, runtime={stats['runtime']:.1f}s")

    if rank == 0:
        local_dir = SCRIPT_DIR / "results" / "local"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_stats = load_all_statistics(local_dir, config)
        hpc_stats = load_all_statistics(stats_dir, config)
        generate_table_csv(local_stats, hpc_stats, config, SCRIPT_DIR / "results" / "table.csv")
        generate_table_latex(local_stats, hpc_stats, config, SCRIPT_DIR / "results" / "table.tex")
        print("\nTable generated.")


if __name__ == "__main__":
    main()
