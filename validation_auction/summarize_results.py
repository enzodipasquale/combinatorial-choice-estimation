#!/bin/env python
"""
Summarize validation results from multiple replications.

This script loads all result files and computes aggregate statistics.
"""

import numpy as np
import json
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Summarize validation results')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: results/)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output summary file (default: results_summary.json)')
    
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else BASE_DIR / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Find all result files
    result_files = sorted(results_dir.glob("results_*.json"))
    
    if len(result_files) == 0:
        print(f"No result files found in {results_dir}")
        return 1
    
    print(f"Found {len(result_files)} result files")
    
    # Load all results
    all_metrics = []
    all_thetas_true = []
    all_thetas_hat = []
    all_times = []
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        all_metrics.append(results["metrics"])
        all_times.append(results["estimation_time"])
        all_thetas_true.append(np.array(results["theta_true"]))
        all_thetas_hat.append(np.array(results["theta_hat"]))
    
    # Convert to arrays
    all_thetas_true = np.array(all_thetas_true)
    all_thetas_hat = np.array(all_thetas_hat)
    
    # Compute aggregate statistics
    summary = {
        "num_replications": len(result_files),
        "metrics": {
            "rmse_mean": np.mean([m["rmse"] for m in all_metrics]),
            "rmse_std": np.std([m["rmse"] for m in all_metrics]),
            "mae_mean": np.mean([m["mae"] for m in all_metrics]),
            "mae_std": np.std([m["mae"] for m in all_metrics]),
            "bias_mean": np.mean([m["bias"] for m in all_metrics]),
            "bias_std": np.std([m["bias_std"] for m in all_metrics]),
            "max_error_mean": np.mean([m["max_error"] for m in all_metrics]),
            "max_error_std": np.std([m["max_error"] for m in all_metrics]),
            "relative_error_mean": np.mean([m["relative_error"] for m in all_metrics]),
            "relative_error_std": np.std([m["relative_error"] for m in all_metrics]),
        },
        "timing": {
            "estimation_time_mean": np.mean(all_times),
            "estimation_time_std": np.std(all_times),
            "estimation_time_min": np.min(all_times),
            "estimation_time_max": np.max(all_times),
        },
        "parameter_recovery": {}
    }
    
    # Per-parameter statistics
    num_features = all_thetas_true.shape[1]
    per_param_rmse = np.sqrt(np.mean((all_thetas_hat - all_thetas_true) ** 2, axis=0))
    per_param_bias = np.mean(all_thetas_hat - all_thetas_true, axis=0)
    per_param_mae = np.mean(np.abs(all_thetas_hat - all_thetas_true), axis=0)
    
    summary["parameter_recovery"] = {
        "per_param_rmse": per_param_rmse.tolist(),
        "per_param_bias": per_param_bias.tolist(),
        "per_param_mae": per_param_mae.tolist(),
        "rmse_mean": np.mean(per_param_rmse),
        "rmse_std": np.std(per_param_rmse),
        "bias_mean": np.mean(per_param_bias),
        "bias_std": np.std(per_param_bias),
    }
    
    # Print summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    print(f"\nNumber of replications: {summary['num_replications']}")
    print(f"\nOverall Metrics:")
    print(f"  RMSE: {summary['metrics']['rmse_mean']:.6f} ± {summary['metrics']['rmse_std']:.6f}")
    print(f"  MAE:  {summary['metrics']['mae_mean']:.6f} ± {summary['metrics']['mae_std']:.6f}")
    print(f"  Bias: {summary['metrics']['bias_mean']:.6f} ± {summary['metrics']['bias_std']:.6f}")
    print(f"  Max Error: {summary['metrics']['max_error_mean']:.6f} ± {summary['metrics']['max_error_std']:.6f}")
    print(f"  Relative Error: {summary['metrics']['relative_error_mean']:.2f}% ± {summary['metrics']['relative_error_std']:.2f}%")
    
    print(f"\nTiming:")
    print(f"  Mean: {summary['timing']['estimation_time_mean']:.2f}s")
    print(f"  Std:  {summary['timing']['estimation_time_std']:.2f}s")
    print(f"  Range: [{summary['timing']['estimation_time_min']:.2f}s, {summary['timing']['estimation_time_max']:.2f}s]")
    
    print(f"\nPer-Parameter Recovery:")
    print(f"  RMSE (mean across params): {summary['parameter_recovery']['rmse_mean']:.6f} ± {summary['parameter_recovery']['rmse_std']:.6f}")
    print(f"  Bias (mean across params):  {summary['parameter_recovery']['bias_mean']:.6f} ± {summary['parameter_recovery']['bias_std']:.6f}")
    
    # Save summary
    output_file = Path(args.output) if args.output else results_dir / "results_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())

