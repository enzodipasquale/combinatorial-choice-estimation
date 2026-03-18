#!/usr/bin/env python3
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper.numerical_experiments.unit_demand.run_experiment import run_replication
from paper.numerical_experiments.unit_demand.compute_statistics import compute_statistics
from paper.numerical_experiments.unit_demand.aggregate_results import (
    load_all_statistics, generate_table_csv, generate_table_latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--dgp", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent
    with open(base / args.config) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    n_reps = args.replications or exp["n_replications"]
    K = exp["K"]
    beta = np.array(exp["beta_star"])
    dgps = [args.dgp] if args.dgp else list(config["dgps"].keys())
    grid_J = config["grid"]["J"]
    grid_N = config["grid"]["N"]

    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dgp in dgps:
        for J, N in product(grid_J, grid_N):
            print(f"\n{'='*60}\n{dgp}, J={J}, N={N}\n{'='*60}")
            results = []
            for rep in range(n_reps):
                print(f"  Rep {rep+1}/{n_reps}...", end=" ", flush=True)
                r = run_replication(dgp, N, J, K, beta, replication=rep, config=config)
                results.append(r)
                print("\u2713")

            stats = compute_statistics(results, N=N)
            with open(results_dir / f"stats_{dgp}_N{N}_J{J}.json", "w") as f:
                json.dump(stats, f, indent=2)

            print(f"  MLE:     RMSE={stats['mle_rmse_mean']:.4f}")
            print(f"  Combest: RMSE={stats['combest_rmse_mean']:.4f}")
            if stats.get("efficiency_ratio_total") is not None:
                print(f"  Efficiency ratio: {stats['efficiency_ratio_total']:.3f}")
            if stats.get("N_mse_mle") is not None:
                print(f"  N*MSE:  MLE={stats['N_mse_mle']:.2f}  Combest={stats['N_mse_combest']:.2f}")

    print(f"\n{'='*60}\nGenerating tables...\n{'='*60}")
    stats = load_all_statistics(results_dir, config)
    generate_table_csv(stats, config, results_dir / "table.csv")
    generate_table_latex(stats, config, results_dir / "table.tex")
    print("Done!")


if __name__ == "__main__":
    main()
