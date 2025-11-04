#!/usr/bin/env python3
"""
Summary script to compute RMSE and Bias per parameter from inversion experiment results.
Computes averages over replications for each method (naive vs IV).
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
        'runtime_median': method_df['time_s'].median(),
        'runtime_min': method_df['time_s'].min(),
        'runtime_max': method_df['time_s'].max(),
    }
    
    # Compute RMSE and Bias for each parameter with standard errors
    for k in range(num_features):
        theta_true_k = method_df[f'theta_true_{k}'].values
        theta_est_k = method_df[f'theta_{k}'].values
        
        # RMSE
        rmse = np.sqrt(np.mean((theta_est_k - theta_true_k) ** 2))
        results[f'rmse_{k}'] = rmse
        
        # Bias (mean error)
        bias = np.mean(theta_est_k - theta_true_k)
        results[f'bias_{k}'] = bias
        
        # Standard deviation of errors (for confidence intervals)
        errors_k = theta_est_k - theta_true_k
        results[f'bias_std_{k}'] = np.std(errors_k)
        
        # Mean absolute error
        mae = np.mean(np.abs(errors_k))
        results[f'mae_{k}'] = mae
        
        # Relative error percentage
        if np.abs(theta_true_k).mean() > 1e-10:
            rel_error = np.mean(np.abs(errors_k) / np.abs(theta_true_k)) * 100
            results[f'rel_error_pct_{k}'] = rel_error
    
    # Timing breakdown statistics
    if 'timing_compute' in method_df.columns:
        results['timing_compute_mean'] = method_df['timing_compute'].mean()
        results['timing_solve_mean'] = method_df['timing_solve'].mean()
        results['timing_comm_mean'] = method_df['timing_comm'].mean()
        results['timing_compute_pct_mean'] = method_df['timing_compute_pct'].mean()
        results['timing_solve_pct_mean'] = method_df['timing_solve_pct'].mean()
        results['timing_comm_pct_mean'] = method_df['timing_comm_pct'].mean()
    
    # Objective value statistics
    if 'obj_value' in method_df.columns:
        obj_vals = method_df['obj_value'].dropna()
        if len(obj_vals) > 0:
            results['obj_value_mean'] = obj_vals.mean()
            results['obj_value_std'] = obj_vals.std()
            results['obj_value_min'] = obj_vals.min()
            results['obj_value_max'] = obj_vals.max()
    
    # OLS and IV regression coefficient metrics with standard errors
    if 'ols_coef_0' in method_df.columns:
        ols_coef = method_df['ols_coef_0'].dropna()
        iv_coef = method_df['iv_coef_0'].dropna()
        ols_se = method_df['ols_se_0'].dropna()
        iv_se = method_df['iv_se_0'].dropna()
        
        if len(ols_coef) > 0:
            results['ols_coef_mean'] = ols_coef.mean()
            results['ols_coef_std'] = ols_coef.std()
            results['ols_coef_median'] = ols_coef.median()
            if len(ols_se) > 0:
                results['ols_se_mean'] = ols_se.mean()
        
        if len(iv_coef) > 0:
            results['iv_coef_mean'] = iv_coef.mean()
            results['iv_coef_std'] = iv_coef.std()
            results['iv_coef_median'] = iv_coef.median()
            if len(iv_se) > 0:
                results['iv_se_mean'] = iv_se.mean()
        
        # Bias reduction from IV (if both available)
        if len(ols_coef) > 0 and len(iv_coef) > 0:
            results['iv_bias_reduction_pct'] = np.mean(np.abs(iv_coef - ols_coef) / (np.abs(ols_coef) + 1e-10)) * 100
    
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
    print(f"Inversion Experiment Summary: {subproblem}")
    print(f"  Agents: {num_agents}, Items: {num_items}, Features: {num_features}")
    print(f"{'='*80}\n")
    
    for _, row in summary_df.iterrows():
        method = row['method']
        runtime = row['runtime_mean']
        print(f"\n{method.upper()} METHOD:")
        print(f"  Runtime: {runtime:.4f} ± {row['runtime_std']:.4f} s (median: {row.get('runtime_median', np.nan):.4f})")
        
        if 'timing_compute_mean' in row:
            print(f"  Timing Breakdown:")
            print(f"    Compute: {row['timing_compute_mean']:.4f}s ({row.get('timing_compute_pct_mean', np.nan):.1f}%)")
            print(f"    Solve: {row['timing_solve_mean']:.4f}s ({row.get('timing_solve_pct_mean', np.nan):.1f}%)")
            print(f"    Comm: {row['timing_comm_mean']:.4f}s ({row.get('timing_comm_pct_mean', np.nan):.1f}%)")
        
        print(f"  RMSE per parameter:")
        for k in range(num_features):
            print(f"    θ_{k}: {row[f'rmse_{k}']:.6f}")
        print(f"  Bias per parameter (with std):")
        for k in range(num_features):
            bias_std = row.get(f'bias_std_{k}', np.nan)
            if not np.isnan(bias_std):
                print(f"    θ_{k}: {row[f'bias_{k}']:+.6f} (± {bias_std:.6f})")
            else:
                print(f"    θ_{k}: {row[f'bias_{k}']:+.6f}")
        
        if 'mae_0' in row:
            print(f"  MAE per parameter:")
            for k in range(num_features):
                print(f"    θ_{k}: {row[f'mae_{k}']:.6f}")
        
        if 'obj_value_mean' in row:
            print(f"  Objective: {row['obj_value_mean']:.4f} ± {row.get('obj_value_std', np.nan):.4f}")
        
        if 'ols_coef_mean' in row:
            print(f"  OLS Coefficient (item feature): {row['ols_coef_mean']:.6f} ± {row['ols_coef_std']:.6f}")
            if 'ols_se_mean' in row:
                print(f"    (Std Error: {row['ols_se_mean']:.6f})")
            print(f"  IV Coefficient (item feature): {row['iv_coef_mean']:.6f} ± {row['iv_coef_std']:.6f}")
            if 'iv_se_mean' in row:
                print(f"    (Std Error: {row['iv_se_mean']:.6f})")
            if 'iv_bias_reduction_pct' in row:
                print(f"  IV Bias Reduction: {row['iv_bias_reduction_pct']:.2f}%")
    
    # Save summary if output path provided
    if output_csv_path:
        summary_df.to_csv(output_csv_path, index=False)
        print(f"\nSummary saved to: {output_csv_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Summarize inversion experiment results')
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


