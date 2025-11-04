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
    'greedy': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{GS}}'],
        'descriptions': '1 agent modular, 1 item modular, 1 gross substitutes'
    },
    'greedy_naive': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{GS}}'],
        'descriptions': '1 agent modular, 1 item modular, 1 gross substitutes'
    },
    'greedy_iv': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{GS}}'],
        'descriptions': '1 agent modular, 1 item modular, 1 gross substitutes'
    },
    'supermod': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD1}}', '\\theta_{\\text{QUAD2}}'],
        'descriptions': '1 agent modular, 1 item modular, 2 quadratic'
    },
    'supermod_naive': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD1}}', '\\theta_{\\text{QUAD2}}'],
        'descriptions': '1 agent modular, 1 item modular, 2 quadratic'
    },
    'supermod_iv': {
        'names': ['\\theta_{\\text{AGENT}}', '\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD1}}', '\\theta_{\\text{QUAD2}}'],
        'descriptions': '1 agent modular, 1 item modular, 2 quadratic'
    },
    'knapsack': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular features'
    },
    'knapsack_naive': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular features'
    },
    'knapsack_iv': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular features'
    },
    'quadknapsack': {
        'names': ['\\theta_{\\text{MOD}}'] * 4 + ['\\theta_{\\text{ITEM}}', '\\theta_{\\text{QUAD}}'],
        'descriptions': '4 agent modular, 1 item modular, 1 quadratic'
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
    'greedy': 'Greedy (Naive vs IV)',
    'greedy_naive': 'Greedy - Naive',
    'greedy_iv': 'Greedy - IV',
    'supermod': 'Supermodular (Naive vs IV)',
    'supermod_naive': 'Supermodular - Naive',
    'supermod_iv': 'Supermodular - IV',
    'knapsack': 'Knapsack (Naive vs IV)',
    'knapsack_naive': 'Knapsack - Naive',
    'knapsack_iv': 'Knapsack - IV',
    'quadknapsack': 'Quad Knapsack (Naive vs IV)',
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
    Generate LaTeX table in 3x3 grid format comparing naive vs IV methods.
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
    
    # Organize sizes into 3x3 grid
    # Extract unique agent and item counts
    agent_counts = sorted(set(n for n, m in results_by_size.keys()))
    item_counts = sorted(set(m for n, m in results_by_size.keys()))
    
    # Ensure we have at least 3x3, pad with None if needed
    while len(agent_counts) < 3:
        agent_counts.append(None)
    while len(item_counts) < 3:
        item_counts.append(None)
    
    # Take first 3 of each
    agent_counts = agent_counts[:3]
    item_counts = item_counts[:3]
    
    # Get all methods
    all_methods = set()
    for df_size in results_by_size.values():
        all_methods.update(df_size['method'].unique())
    methods = sorted([m for m in ['naive', 'iv'] if m in all_methods])
    
    # Helper function to extract parameter label from LaTeX string
    def extract_param_label(param_latex):
        """Extract parameter label like 'MOD', 'GS', 'QUAD1', 'QUAD2', 'AGENT', 'ITEM' from LaTeX string."""
        # Handle specific cases
        if 'QUAD1' in param_latex:
            return 'QUAD1'
        elif 'QUAD2' in param_latex:
            return 'QUAD2'
        elif 'AGENT' in param_latex:
            return 'AGENT'
        elif 'ITEM' in param_latex:
            return 'ITEM'
        elif 'GS' in param_latex:
            return 'GS'
        elif 'QUAD' in param_latex:
            return 'QUAD'
        elif 'MOD' in param_latex:
            return 'MOD'
        # Fallback: extract from structure
        s = param_latex.replace('\\', '').replace('{', '').replace('}', '').replace('theta', '').replace('text', '').strip()
        return s if s else 'MOD'
    
    # Helper function to get value for a (N, M) combination
    def get_value(method, n, m, metric_type='runtime'):
        """Get value for a specific method, size, and metric."""
        key = (n, m)
        if key not in results_by_size:
            return "---"
        
        df_size = results_by_size[key]
        df_method = df_size[df_size['method'] == method]
        
        if len(df_method) == 0:
            return "---"
        
        if metric_type == 'runtime':
            runtime_mean = df_method['time_s'].mean()
            runtime_std = df_method['time_s'].std()
            return f"${format_number(runtime_mean, 3)} \\pm {format_number(runtime_std, 3)}$"
        else:
            aggregated = aggregate_parameters_by_type(df_method, param_config)
            # metric_type is like 'rmse_MOD' or 'bias_GS'
            parts = metric_type.split('_', 1)
            if len(parts) != 2:
                return "---"
            metric, param_label = parts
            # Find matching param type in aggregated
            for param_type in aggregated.keys():
                if extract_param_label(param_type) == param_label:
                    idx = 0 if metric == 'rmse' else 1
                    return f"${format_number(aggregated[param_type][idx], 4)}$"
            return "---"
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{Inversion Experiment: {EXPERIMENT_TITLES.get(experiment_type, experiment_type)} (3$\\times$3 Grid: $N \\times M$)}}")
    lines.append(f"\\label{{tab:inversion_{experiment_type}}}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{l c c c c c c c c c}")
    lines.append("\\toprule")
    
    # Header: M (Items) across top
    header1 = "Method & \\multicolumn{9}{c}{$M$ (Items)} \\\\"
    lines.append(header1)
    lines.append("\\cmidrule(lr){2-10}")
    
    # Header row 2: Item values
    header2 = " & \\multicolumn{3}{c}{" + str(item_counts[0]) + "} & \\multicolumn{3}{c}{" + str(item_counts[1]) + "} & \\multicolumn{3}{c}{" + str(item_counts[2]) + "} \\\\"
    lines.append(header2)
    lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
    
    # Header row 3: Agent values
    header3 = " & $N=" + str(agent_counts[0]) + "$ & $N=" + str(agent_counts[1]) + "$ & $N=" + str(agent_counts[2]) + "$"
    header3 += " & $N=" + str(agent_counts[0]) + "$ & $N=" + str(agent_counts[1]) + "$ & $N=" + str(agent_counts[2]) + "$"
    header3 += " & $N=" + str(agent_counts[0]) + "$ & $N=" + str(agent_counts[1]) + "$ & $N=" + str(agent_counts[2]) + "$ \\\\"
    lines.append(header3)
    lines.append("\\midrule")
    
    # Generate table: one section per metric type
    # 1. Runtime
    lines.append("\\multicolumn{10}{l}{\\textit{Runtime (s)}} \\\\")
    for method in methods:
        row_parts = [METHOD_NAMES.get(method, method.title())]
        for m in item_counts:
            for n in agent_counts:
                if n is None or m is None:
                    row_parts.append("---")
                else:
                    row_parts.append(get_value(method, n, m, 'runtime'))
        lines.append(" & ".join(row_parts) + " \\\\")
    lines.append("\\cmidrule(lr){1-10}")
    
    # 2. RMSE for each parameter type
    for param_type in unique_params:
        param_label = extract_param_label(param_type)
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{RMSE}} ${param_type}$}} \\\\")
        for method in methods:
            row_parts = [METHOD_NAMES.get(method, method.title())]
            for m in item_counts:
                for n in agent_counts:
                    if n is None or m is None:
                        row_parts.append("---")
                    else:
                        row_parts.append(get_value(method, n, m, f'rmse_{param_label}'))
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\cmidrule(lr){1-10}")
    
    # 3. Bias for each parameter type
    for param_type in unique_params:
        param_label = extract_param_label(param_type)
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{Bias}} ${param_type}$}} \\\\")
        for method in methods:
            row_parts = [METHOD_NAMES.get(method, method.title())]
            for m in item_counts:
                for n in agent_counts:
                    if n is None or m is None:
                        row_parts.append("---")
                    else:
                        row_parts.append(get_value(method, n, m, f'bias_{param_label}'))
            lines.append(" & ".join(row_parts) + " \\\\")
        if param_type != unique_params[-1]:
            lines.append("\\cmidrule(lr){1-10}")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\footnotesize")
    available_sizes = sorted(results_by_size.keys())
    size_str = ", ".join([f"({n},{m})" for n, m in available_sizes])
    # Count replications
    if len(available_sizes) > 0:
        first_size_df = results_by_size[available_sizes[0]]
        num_replications = len(first_size_df[first_size_df['method'] == first_size_df['method'].iloc[0]]) if len(first_size_df) > 0 else 0
        lines.append(f"\\item Results from {num_replications} replications per size.")
    lines.append(f"\\item Runtime reported as mean $\\pm$ standard deviation in seconds. Each cell shows $(N,M)$ combination.")
    lines.append(f"\\item Data available for $(N,M) \\in \\{{{size_str}\\}}$.")
    missing = []
    for m in item_counts:
        for n in agent_counts:
            if n is not None and m is not None and (n, m) not in results_by_size:
                missing.append(f"({n},{m})")
    if missing:
        lines.append(f"\\item Missing combinations: {', '.join(missing)}")
    lines.append("\\item RMSE: Root Mean Squared Error. Bias: mean estimation error.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    lines.append("")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated LaTeX table: {output_path}")
    print(f"  - 3x3 grid: {len(agent_counts)} agents x {len(item_counts)} items")
    print(f"  - Available sizes: {available_sizes}")
    print(f"  - {len(unique_params)} unique parameter types")


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from inversion experiment results')
    parser.add_argument('results_csv', type=str, nargs='?', default=None,
                       help='Path to results CSV file (or base directory if using --experiment)')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['greedy', 'greedy_naive', 'greedy_iv', 'supermod', 'supermod_naive', 'supermod_iv',
                                'knapsack', 'knapsack_naive', 'knapsack_iv', 'quadknapsack', 'quadknapsack_naive', 'quadknapsack_iv'],
                       help='Experiment type (used if results_csv is a directory)')
    parser.add_argument('--output', '-o', type=str, default='tables.tex',
                       help='Output LaTeX file path')
    
    args = parser.parse_args()
    
    # Determine results path and experiment type
    if args.results_csv is None and args.experiment is None:
        print("Error: Must provide either results_csv path or --experiment")
        return 1
    
    if args.results_csv and Path(args.results_csv).is_file():
        # Direct CSV file path
        results_path = Path(args.results_csv)
        # Determine experiment type from CSV or use provided
        if args.experiment:
            exp_type = args.experiment
        else:
            # Try to infer from CSV - read first row to get subproblem
            df = pd.read_csv(results_path, nrows=1)
            if 'subproblem' in df.columns:
                subproblem = df['subproblem'].iloc[0]
                # Map subproblem to experiment type
                exp_type_map = {
                    'Greedy': 'greedy',
                    'QuadSupermodularNetwork': 'supermod',
                    'LinearKnapsack': 'knapsack',
                    'QuadKnapsack': 'quadknapsack'
                }
                exp_type = exp_type_map.get(subproblem, 'greedy')
            else:
                exp_type = 'greedy'  # Default
    else:
        # Directory path - use old style
        base_dir = Path(args.results_csv) if args.results_csv else Path('.')
        if not args.experiment:
            print("Error: Must provide --experiment when results_csv is a directory")
            return 1
        
        exp_dir = base_dir / args.experiment
        exp_type = args.experiment
        
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
    
    # Determine experiment type for parameter config
    # If base name (greedy, supermod, etc.), check CSV for methods
    if exp_type in ['greedy', 'supermod', 'knapsack', 'quadknapsack']:
        # Check if CSV has both naive and IV methods
        df_sample = pd.read_csv(results_path, nrows=100)
        if 'method' in df_sample.columns:
            methods_in_csv = df_sample['method'].dropna().unique()
            # Filter out ERROR
            methods_in_csv = [m for m in methods_in_csv if m != 'ERROR']
            if 'naive' in methods_in_csv and 'iv' in methods_in_csv:
                # Use base type - table generator will handle both methods
                exp_type_for_params = exp_type
            elif 'naive' in methods_in_csv:
                exp_type_for_params = f"{exp_type}_naive"
            elif 'iv' in methods_in_csv:
                exp_type_for_params = f"{exp_type}_iv"
            else:
                exp_type_for_params = exp_type
        else:
            exp_type_for_params = exp_type
    else:
        exp_type_for_params = exp_type
    
    output_path = Path(args.output)
    generate_table(exp_type_for_params, str(results_path), str(output_path))
    
    return 0


if __name__ == '__main__':
    exit(main())


