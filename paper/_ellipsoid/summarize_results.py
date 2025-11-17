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
        theta_true_col = f'theta_true_{k}'
        theta_est_col = f'theta_{k}'
        
        # Check if columns exist
        if theta_true_col not in method_df.columns or theta_est_col not in method_df.columns:
            print(f"Warning: Missing theta columns for k={k}, skipping")
            results[f'rmse_{k}'] = np.nan
            results[f'bias_{k}'] = np.nan
            continue
        
        theta_true_k = pd.to_numeric(method_df[theta_true_col], errors='coerce').values
        theta_est_k = pd.to_numeric(method_df[theta_est_col], errors='coerce').values
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(theta_true_k) | np.isnan(theta_est_k))
        if valid_mask.sum() == 0:
            results[f'rmse_{k}'] = np.nan
            results[f'bias_{k}'] = np.nan
            continue
        
        theta_true_k = theta_true_k[valid_mask]
        theta_est_k = theta_est_k[valid_mask]
        
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
    # Read CSV with error handling for malformed rows
    df = pd.read_csv(results_csv_path, on_bad_lines='skip', engine='python')
    
    # Filter out ERROR rows
    df = df[df['method'] != 'ERROR'].copy()
    
    if len(df) == 0:
        print(f"No valid results found in {results_csv_path}")
        return None
    
    # Validate that required columns exist
    required_cols = ['num_features', 'num_agents', 'num_items', 'subproblem']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Get experiment metadata (should be same across rows)
    # Convert to numeric, handling any string values
    try:
        num_features = int(pd.to_numeric(df['num_features'].iloc[0], errors='coerce'))
        num_agents = int(pd.to_numeric(df['num_agents'].iloc[0], errors='coerce'))
        num_items = int(pd.to_numeric(df['num_items'].iloc[0], errors='coerce'))
        subproblem = str(df['subproblem'].iloc[0])
    except (ValueError, TypeError) as e:
        print(f"Error reading metadata from CSV: {e}")
        print(f"First row values: num_features={df['num_features'].iloc[0]}, num_agents={df['num_agents'].iloc[0]}, num_items={df['num_items'].iloc[0]}")
        return None
    
    if pd.isna(num_features) or pd.isna(num_agents) or pd.isna(num_items):
        print(f"Error: Invalid metadata values (NaN detected)")
        return None
    
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
    
    # Compute objective consistency statistics (across all replications)
    df_valid = df[df['method'] != 'ERROR'].copy()
    if len(df_valid) > 0 and 'obj_diff_rg_1slack' in df_valid.columns:
        # Group by replication to get unique objective differences
        replications = df_valid['replication'].unique()
        obj_diffs = df_valid.groupby('replication').first()  # All methods in same replication have same diff values
        obj_close_pct = (obj_diffs['obj_close_all'].sum() / len(obj_diffs)) * 100 if 'obj_close_all' in obj_diffs.columns else 0
        obj_diff_rg_1slack_mean = obj_diffs['obj_diff_rg_1slack'].mean() if 'obj_diff_rg_1slack' in obj_diffs.columns else np.nan
        obj_diff_rg_ellipsoid_mean = obj_diffs['obj_diff_rg_ellipsoid'].mean() if 'obj_diff_rg_ellipsoid' in obj_diffs.columns else np.nan
        obj_diff_1slack_ellipsoid_mean = obj_diffs['obj_diff_1slack_ellipsoid'].mean() if 'obj_diff_1slack_ellipsoid' in obj_diffs.columns else np.nan
    else:
        obj_close_pct = 0
        obj_diff_rg_1slack_mean = np.nan
        obj_diff_rg_ellipsoid_mean = np.nan
        obj_diff_1slack_ellipsoid_mean = np.nan
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"Experiment Summary: {subproblem}")
    print(f"  Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
    print(f"{'='*80}\n")
    
    # Print objective consistency summary
    if not np.isnan(obj_diff_rg_1slack_mean):
        print(f"Objective Consistency (across all methods):")
        print(f"  All methods close (within 1e-3): {obj_close_pct:.1f}% of replications")
        print(f"  Mean |obj_rg - obj_1slack|: {obj_diff_rg_1slack_mean:.6e}")
        print(f"  Mean |obj_rg - obj_ellipsoid|: {obj_diff_rg_ellipsoid_mean:.6e}")
        print(f"  Mean |obj_1slack - obj_ellipsoid|: {obj_diff_1slack_ellipsoid_mean:.6e}")
        print()
    
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


