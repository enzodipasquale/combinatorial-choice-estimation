#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path


def compute_statistics(results, N=None):
    if not results:
        return {}

    beta_star = np.array(results[0]["beta_star"])
    K = len(beta_star)
    mle_hats = np.array([r["beta_mle"] for r in results])
    combest_hats = np.array([r["beta_combest"] for r in results])

    def stats(hats):
        errors = hats - beta_star
        bias = errors.mean(axis=0)           # (K,)
        var = errors.var(axis=0, ddof=1)     # (K,)
        mse = (errors**2).mean(axis=0)       # (K,)
        return {
            "bias_mean": float(bias.mean()),
            "rmse_mean": float(np.sqrt(mse).mean()),
            "mse_per_coef": mse.tolist(),
            "var_per_coef": var.tolist(),
            "mse_total": float(mse.sum()),
            "var_total": float(var.sum()),
        }

    s_mle = stats(mle_hats)
    s_combest = stats(combest_hats)

    out = {
        "n_replications": len(results),
        "K": K,
    }
    for key, val in s_mle.items():
        out[f"mle_{key}"] = val
    for key, val in s_combest.items():
        out[f"combest_{key}"] = val

    # Efficiency ratio: MSE_combest / MSE_mle (per coef and total)
    mse_mle = np.array(s_mle["mse_per_coef"])
    mse_combest = np.array(s_combest["mse_per_coef"])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_per_coef = np.where(mse_mle > 0, mse_combest / mse_mle, np.nan)
    out["efficiency_ratio_per_coef"] = ratio_per_coef.tolist()
    out["efficiency_ratio_total"] = (
        float(s_combest["mse_total"] / s_mle["mse_total"])
        if s_mle["mse_total"] > 0 else None
    )

    # N * MSE (should stabilize as N grows if sqrt-N consistent)
    if N is not None:
        out["N"] = N
        out["N_mse_mle"] = float(N * s_mle["mse_total"])
        out["N_mse_combest"] = float(N * s_combest["mse_total"])

    out["runtime_mle"] = float(np.mean([r["runtime_mle"] for r in results]))
    out["runtime_combest"] = float(np.mean([r["runtime_combest"] for r in results]))

    return out


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
    stats = compute_statistics(results, N=args.N)

    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stats_{args.dgp}_N{args.N}_J{args.J}.json"

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"{args.dgp}, N={args.N}, J={args.J}: {stats['n_replications']} reps")
    for method in ("mle", "combest"):
        print(f"  {method}: Bias={stats[f'{method}_bias_mean']:.4f}, "
              f"RMSE={stats[f'{method}_rmse_mean']:.4f}")
    if "efficiency_ratio_total" in stats:
        print(f"  Efficiency ratio (MSE_combest/MSE_mle): {stats['efficiency_ratio_total']:.4f}")


if __name__ == "__main__":
    main()
