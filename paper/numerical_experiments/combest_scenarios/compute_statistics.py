#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path


def load_replication_results(results_dir, spec, N, M):
    results = []
    for f in sorted(results_dir.glob(f"{spec}_N{N}_M{M}_rep*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def compute_statistics(results):
    if not results:
        return {k: np.nan for k in [
            "bias_alpha", "rmse_alpha", "bias_lambda", "rmse_lambda", "runtime"]}

    theta_hats = np.array([r["theta_hat"] for r in results])
    theta_stars = np.array([r["theta_star"] for r in results])

    alpha_idx = np.array(results[0].get("alpha_indices", []))
    lambda_idx = np.array(results[0].get("lambda_indices", []))
    BOUND_MARGIN = 99.0

    def param_stats(idx):
        if len(idx) == 0:
            return np.nan, np.nan
        p_hat, p_star = theta_hats[:, idx], theta_stars[:, idx]
        interior = np.abs(p_hat) < BOUND_MARGIN
        bias = rmse = np.nan
        if np.any(interior):
            errors = np.where(interior, p_hat - p_star, np.nan)
            bias = float(np.nanmean(errors))
            rmse = float(np.sqrt(np.nanmean(np.where(interior, errors**2, np.nan))))
        return bias, rmse

    bias_alpha, rmse_alpha = param_stats(alpha_idx)
    bias_lambda, rmse_lambda = param_stats(lambda_idx)

    runtime = float(np.mean([r["runtime"] for r in results]))

    return {
        "bias_alpha": float(bias_alpha),
        "rmse_alpha": float(rmse_alpha),
        "bias_lambda": float(bias_lambda),
        "rmse_lambda": float(rmse_lambda),
        "runtime": runtime,
        "n_replications": len(results),
        "avg_at_bound": float(np.mean([r.get("n_at_bound", 0) for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--results-dir", type=str, default="results/raw")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / args.results_dir
    results = load_replication_results(results_dir, args.spec, args.N, args.M)
    stats = compute_statistics(results)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / (args.output_dir or "results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"stats_{args.spec}_N{args.N}_M{args.M}.json"

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics for {args.spec}, N={args.N}, M={args.M}:")
    for label, key in [("Bias α", "bias_alpha"), ("RMSE α", "rmse_alpha"),
                        ("Bias λ", "bias_lambda"), ("RMSE λ", "rmse_lambda")]:
        print(f"  {label}: {stats[key]:.6f}")
    print(f"  Runtime: {stats['runtime']:.2f}s")
    print(f"  Replications: {stats['n_replications']}")
    print(f"  Avg params at bound: {stats['avg_at_bound']:.1f}")


if __name__ == "__main__":
    main()
