#!/usr/bin/env python3
"""
Generate LaTeX tables from inversion experiment results for paper submission.
Compares naive (biased) vs IV (unbiased) estimation methods across multiple sizes.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


# Parameter naming for different experiment types
PARAM_NAMES = {
    'greedy_naive': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{GS}}'],
        'descriptions': '1 agent modular, 1 item modular, 1 gross substitutes'
    },
    'greedy_iv': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{GS}}'],
        'descriptions': '1 agent modular, 1 item modular, 1 gross substitutes'
    },
    'supermod_naive': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD1}}', '\\theta_{\\text{QUAD2}}'],
        'descriptions': '1 agent modular, 1 item modular, 2 quadratic'
    },
    'supermod_iv': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD1}}', '\\theta_{\\text{QUAD2}}'],
        'descriptions': '1 agent modular, 1 item modular, 2 quadratic'
    },
    'knapsack_naive': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular features'
    },
    'knapsack_iv': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular features'
    },
    'quadknapsack_naive': {
        'names': ['\\theta_{\\text{MOD}}'] * 4 + ['\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD}}'],
        'descriptions': '4 agent modular, 1 item modular, 1 quadratic'
    },
    'quadknapsack_iv': {
        'names': ['\\theta_{\\text{MOD}}'] * 4 + ['\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD}}'],
        'descriptions': '4 agent modular, 1 item modular, 1 quadratic'
    }
}

# Method display names
METHOD_NAMES = {
    'naive': 'Naive (Biased)',
    'iv': 'IV (Unbiased)'
}

# Experiment type titles
EXPERIMENT_TITLES = {
    'greedy_naive': 'Greedy - Naive',
    'greedy_iv': 'Greedy - IV',
    'supermod_naive': 'Supermodular - Naive',
    'supermod_iv': 'Supermodular - IV',
    'knapsack_naive': 'Knapsack - Naive',
    'knapsack_iv': 'Knapsack - IV',
    'quadknapsack_naive': 'Quad Knapsack - Naive',
    'quadknapsack_iv': 'Quad Knapsack - IV'
}


def format_number(x, precision=4):
    """Format number for LaTeX table."""
    if np.isnan(x) or np.isinf(x):
        return '---'
    if abs(x) < 1e-10:
        return '0.0'
    if abs(x) < 0.001:
        return f'{x:.2e}'
    if abs(x) < 1:
        return f'{x:.{precision}f}'
    if abs(x) < 100:
        return f'{x:.{precision-1}f}'
    return f'{x:.{precision-2}f}'


def compute_aggregate_rmse_bias(df_group, param_idx):
    """Compute RMSE and Bias for a parameter across all replications."""
    theta_true = df_group[f'theta_true_{param_idx}'].values
    theta_est = df_group[f'theta_{param_idx}'].values
    
    rmse = np.sqrt(np.mean((theta_est - theta_true) ** 2))
    bias = np.mean(theta_est - theta_true)
    
    return rmse, bias


def aggregate_parameters_by_type(df_method, param_config):
    """
    Aggregate parameters by type (e.g., all theta_MOD together).
    Returns dict mapping parameter type -> (rmse, bias)
    """
    param_names = param_config['names']
    num_params = len(param_names)
    
    # Group parameters by type
    param_groups = {}
    for k, param_name in enumerate(param_names):
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(k)
    
    # Compute aggregate RMSE and Bias for each group
    aggregated = {}
    for param_type, indices in param_groups.items():
        rmse_list = []
        bias_list = []
        for k in indices:
            if f'theta_true_{k}' in df_method.columns and f'theta_{k}' in df_method.columns:
                rmse, bias = compute_aggregate_rmse_bias(df_method, k)
                rmse_list.append(rmse)
                bias_list.append(bias)
        if rmse_list:
            aggregated[param_type] = (np.mean(rmse_list), np.mean(bias_list))
    
    return aggregated


def load_results_from_csv(results_csv_path):
    """
    Load results CSV and group by size (num_agents, num_items).
    Returns dict mapping (num_agents, num_items) -> DataFrame
    """
    df = pd.read_csv(results_csv_path)
    df = df[df['method'] != 'ERROR'].copy()
    
    if len(df) == 0:
        return {}
    
    # Group by size
    results_by_size = {}
    for (num_agents, num_items), size_df in df.groupby(['num_agents', 'num_items']):
        key = (int(num_agents), int(num_items))
        results_by_size[key] = size_df
    
    return results_by_size


def generate_table(experiment_type, results_csv_path, output_path):
    """
    Generate LaTeX table comparing naive vs IV methods across multiple sizes.
    Professional formatting for top-tier journals (Econometrica, etc.)
    """
    results_by_size = load_results_from_csv(results_csv_path)
    
    if len(results_by_size) == 0:
        print(f"No valid results found in {results_csv_path}")
        return
    
    # Get param config
    param_config = PARAM_NAMES.get(experiment_type, {
        'names': ['\\theta_0', '\\theta_1'],
        'descriptions': 'parameters'
    })
    
    param_names = param_config['names']
    unique_params = sorted(set(param_names), key=lambda x: param_names.index(x))
    num_unique = len(unique_params)
    
    # Sort sizes by num_agents, then num_items
    sorted_sizes = sorted(results_by_size.keys(), key=lambda x: (x[0], x[1]))
    
    # Get all methods present across all sizes
    all_methods = set()
    for df_size in results_by_size.values():
        all_methods.update(df_size['method'].unique())
    methods = sorted([m for m in ['naive', 'iv'] if m in all_methods])
    
    # Determine number of columns needed per size
    num_cols_per_size = 1 + (2 * num_unique)  # Runtime + (RMSE + Bias) * unique_params
    total_cols = 1 + len(sorted_sizes) * num_cols_per_size
    
    # Use landscape if table is very wide (more than 30 columns)
    use_landscape = total_cols > 30
    
    # Build table content
    lines = []
    lines.append("\\begin{table}[htbp]")
    if use_landscape:
        lines.insert(-1, "\\begin{landscape}")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{Inversion Experiment Results: {EXPERIMENT_TITLES.get(experiment_type, experiment_type)}}}")
    lines.append(f"\\label{{tab:inversion_{experiment_type}}}")
    lines.append("\\begin{threeparttable}")
    
    # Build column specification
    colspec = "l "  # Method column
    for _ in sorted_sizes:
        colspec += "r " + "r" * num_unique + " " + "r" * num_unique + " "  # Runtime + RMSE params + Bias params
    
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    
    # Header row 1: Size labels (show both I and J)
    header1 = "Method"
    for num_agents, num_items in sorted_sizes:
        header1 += f" & \\multicolumn{{{num_cols_per_size}}}{{c}}{{$(I,J)=({num_agents},{num_items})$}}"
    header1 += " \\\\"
    lines.append(header1)
    
    # Header row 2: Column types
    header2 = " & "
    for num_agents, num_items in sorted_sizes:
        header2 += f"Runtime (s) & \\multicolumn{{{num_unique}}}{{c}}{{RMSE}} & \\multicolumn{{{num_unique}}}{{c}}{{Bias}} & "
    header2 = header2.rstrip(" & ") + " \\\\"
    lines.append(header2)
    
    # Header row 3: Parameter names
    header3 = " & "
    for num_agents, num_items in sorted_sizes:
        param_cols = " & ".join([f"${p}$" for p in unique_params])
        header3 += f" & {param_cols} & {param_cols} & "
    header3 = header3.rstrip(" & ") + " \\\\"
    lines.append(header3)
    
    # Cmidrules
    cmidrule = f"\\cmidrule(lr){{2-{1+num_cols_per_size}}}"
    for i in range(1, len(sorted_sizes)):
        start = 2 + i * num_cols_per_size
        end = start + num_cols_per_size - 1
        cmidrule += f" \\cmidrule(lr){{{start}-{end}}}"
    lines.append(cmidrule)
    
    lines.append("\\midrule")
    
    # Generate rows for each method
    for method in methods:
        row_parts = [METHOD_NAMES.get(method, method.title())]
        
        for num_agents, num_items in sorted_sizes:
            df_size = results_by_size[(num_agents, num_items)]
            df_method = df_size[df_size['method'] == method]
            
            if len(df_method) == 0:
                # No data for this method/size
                row_parts.extend(["---"] + ["---"] * num_unique + ["---"] * num_unique)
                continue
            
            # Runtime
            runtime_mean = df_method['time_s'].mean()
            runtime_std = df_method['time_s'].std()
            row_parts.append(f"${format_number(runtime_mean, 3)} \\pm {format_number(runtime_std, 3)}$")
            
            # Aggregate parameters by type
            aggregated = aggregate_parameters_by_type(df_method, param_config)
            
            # Add RMSE and Bias for each unique parameter type
            for param_type in unique_params:
                if param_type in aggregated:
                    row_parts.append(f"${format_number(aggregated[param_type][0], 4)}$")
                else:
                    row_parts.append("---")
            
            for param_type in unique_params:
                if param_type in aggregated:
                    row_parts.append(f"${format_number(aggregated[param_type][1], 4)}$")
                else:
                    row_parts.append("---")
        
        row = " & ".join(row_parts) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    # Add regression coefficients row if available
    has_regression = False
    regression_row = " & Regression:"
    for num_agents, num_items in sorted_sizes:
        df_size = results_by_size[(num_agents, num_items)]
        df_iv = df_size[df_size['method'] == 'iv']
        if len(df_iv) > 0 and 'ols_coef_0' in df_iv.columns:
            ols_coef = df_iv['ols_coef_0'].dropna().mean()
            iv_coef = df_iv['iv_coef_0'].dropna().mean()
            ols_se = df_iv['ols_se_0'].dropna().mean() if 'ols_se_0' in df_iv.columns else np.nan
            iv_se = df_iv['iv_se_0'].dropna().mean() if 'iv_se_0' in df_iv.columns else np.nan
            
            if not np.isnan(ols_se):
                regression_row += f" & OLS: ${format_number(ols_coef, 4)}$, IV: ${format_number(iv_coef, 4)}$"
                regression_row += f" \\\\ \\textit{{(SE)}} & "
                for _ in range(num_cols_per_size - 1):
                    regression_row += " & "
                regression_row += f" & OLS: $({format_number(ols_se, 4)})$, IV: $({format_number(iv_se, 4)})$"
                has_regression = True
                break
    
    if has_regression:
        lines.append("\\midrule")
        lines.append(regression_row + " \\\\")
    
    lines.append("\\end{threeparttable}")
    lines.append("\\begin{tablenotes}")
    lines.append(f"\\footnotesize")
    
    # Count replications
    first_size_df = list(results_by_size.values())[0]
    num_replications = len(first_size_df[first_size_df['method'] == first_size_df['method'].iloc[0]])
    
    lines.append(f"\\item Results from {num_replications} replications per size.")
    lines.append(f"\\item Runtime reported as mean $\\pm$ standard deviation in seconds.")
    lines.append(f"\\item RMSE: Root Mean Squared Error. Bias: mean estimation error.")
    if has_regression:
        lines.append(f"\\item OLS and IV coefficients from second-stage regression of inverted $\\delta$ on item features.")
        lines.append(f"\\item Standard errors (SE) in parentheses.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    if use_landscape:
        lines.append("\\end{landscape}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated LaTeX table: {output_path}")
    print(f"  - {len(sorted_sizes)} sizes: {sorted_sizes}")
    print(f"  - {num_replications} replications per size")
    print(f"  - {num_unique} unique parameter types")


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from inversion experiment results')
    parser.add_argument('base_dir', type=str, help='Base directory containing experiment folders')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['greedy_naive', 'greedy_iv', 'supermod_naive', 'supermod_iv',
                                'knapsack_naive', 'knapsack_iv', 'quadknapsack_naive', 'quadknapsack_iv'],
                       help='Experiment type')
    parser.add_argument('--output', '-o', type=str, default='tables.tex',
                       help='Output LaTeX file path')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    exp_dir = base_dir / args.experiment
    
    # Find results file
    import yaml
    config_path = exp_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        results_csv = cfg.get('results_csv', 'results.csv')
    else:
        results_csv = 'results.csv'
    
    results_path = exp_dir / results_csv
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    output_path = Path(args.output)
    generate_table(args.experiment, str(results_path), str(output_path))
    
    return 0


if __name__ == '__main__':
    exit(main())


