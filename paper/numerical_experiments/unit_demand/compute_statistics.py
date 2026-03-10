#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path


def compute_statistics(results):
    if not results:
        return {}

    beta_star = np.array(results[0]["beta_star"])
    mle_hats = np.array([r["beta_mle"] for r in results])
    combest_hats = np.array([r["beta_combest"] for r in results])

    def stats(hats):
        errors = hats - beta_star
        bias = float(errors.mean(axis=0).mean())
        rmse = float(np.sqrt((errors**2).mean(axis=0)).mean())
        return bias, rmse

    bias_mle, rmse_mle = stats(mle_hats)
    bias_combest, rmse_combest = stats(combest_hats)

    return {
        "bias_mle": bias_mle,
        "rmse_mle": rmse_mle,
        "bias_combest": bias_combest,
        "rmse_combest": rmse_combest,
        "runtime_mle": float(np.mean([r["runtime_mle"] for r in results])),
        "runtime_combest": float(np.mean([r["runtime_combest"] for r in results])),
        "n_replications": len(results),
    }


def load_replication_results(results_dir, dgp, N, J):
    results = []
    for f in sorted(results_dir.glob(f"{dgp}_N{N}_J{J}_rep*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", required=True, choices=["logit", "probit"])
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--J", type=int, required=True)
    parser.add_argument("--results-dir", type=str, default="results/raw")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    base = Path(__file__).parent
    results = load_replication_results(base / args.results_dir, args.dgp, args.N, args.J)
    stats = compute_statistics(results)

    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stats_{args.dgp}_N{args.N}_J{args.J}.json"

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"{args.dgp}, N={args.N}, J={args.J}: {stats['n_replications']} reps")
    for method in ("mle", "combest"):
        print(f"  {method}: Bias={stats[f'bias_{method}']:.4f}, "
              f"RMSE={stats[f'rmse_{method}']:.4f}, Time={stats[f'runtime_{method}']:.2f}s")


if __name__ == "__main__":
    main()
