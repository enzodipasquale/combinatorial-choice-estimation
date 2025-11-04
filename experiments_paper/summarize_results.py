#!/usr/bin/env python3
"""
Summary script to compute RMSE and Bias per parameter from experiment results.
Computes averages over replications for each method.
"""
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def compute_metrics(df, method_name, num_features):
    """Compute RMSE and Bias for each parameter for a given method."""
    method_df = df[df['method'] == method_name].copy()
    
    if len(method_df) == 0:
        return None
    
    results = {
        'method': method_name,
        'runtime_mean': method_df['time_s'].mean(),
        'runtime_std': method_df['time_s'].std(),
    }
    
    # Compute RMSE and Bias for each parameter
    for k in range(num_features):
        theta_true_k = method_df[f'theta_true_{k}'].values
        theta_est_k = method_df[f'theta_{k}'].values
        
        # RMSE
        rmse = np.sqrt(np.mean((theta_est_k - theta_true_k) ** 2))
        results[f'rmse_{k}'] = rmse
        
        # Bias (mean error)
        bias = np.mean(theta_est_k - theta_true_k)
        results[f'bias_{k}'] = bias
    
    return results


def summarize_experiment(results_csv_path, output_csv_path=None):
    """
    Summarize experiment results from CSV file.
    
    Args:
        results_csv_path: Path to results CSV file
        output_csv_path: Optional path to save summary CSV
    """
    df = pd.read_csv(results_csv_path)
    
    # Filter out ERROR rows
    df = df[df['method'] != 'ERROR'].copy()
    
    if len(df) == 0:
        print(f"No valid results found in {results_csv_path}")
        return None
    
    # Get experiment metadata (should be same across rows)
    num_features = int(df['num_features'].iloc[0])
    num_agents = int(df['num_agents'].iloc[0])
    num_items = int(df['num_items'].iloc[0])
    subproblem = df['subproblem'].iloc[0]
    
    # Get unique methods
    methods = df['method'].unique()
    
    # Compute metrics for each method
    summary_rows = []
    for method in methods:
        metrics = compute_metrics(df, method, num_features)
        if metrics is not None:
            metrics['num_agents'] = num_agents
            metrics['num_items'] = num_items
            metrics['num_features'] = num_features
            metrics['subproblem'] = subproblem
            metrics['num_replications'] = len(df[df['method'] == method])
            summary_rows.append(metrics)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"Experiment Summary: {subproblem}")
    print(f"  Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
    print(f"{'='*80}\n")
    
    for _, row in summary_df.iterrows():
        method = row['method']
        runtime = row['runtime_mean']
        print(f"\n{method.upper()}:")
        print(f"  Runtime: {runtime:.4f} ± {row['runtime_std']:.4f} s")
        print(f"  RMSE per parameter:")
        for k in range(num_features):
            print(f"    θ_{k}: {row[f'rmse_{k}']:.6f}")
        print(f"  Bias per parameter:")
        for k in range(num_features):
            print(f"    θ_{k}: {row[f'bias_{k}']:+.6f}")
    
    # Save summary if output path provided
    if output_csv_path:
        summary_df.to_csv(output_csv_path, index=False)
        print(f"\nSummary saved to: {output_csv_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results')
    parser.add_argument('results_csv', type=str, help='Path to results CSV file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output CSV path for summary (default: results_summary.csv in same directory)')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    # Default output path
    if args.output is None:
        output_path = results_path.parent / 'results_summary.csv'
    else:
        output_path = Path(args.output)
    
    summarize_experiment(str(results_path), str(output_path))
    
    return 0


if __name__ == '__main__':
    exit(main())


