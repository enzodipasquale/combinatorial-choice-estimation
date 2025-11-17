#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results for paper submission.

This script creates professional LaTeX tables comparing different estimation methods
across various problem sizes and settings.
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


# Parameter naming for different experiment types
PARAM_NAMES = {
    'Greedy': {
        'names': ['\\theta_{\\text{MOD}}'] * 4 + ['\\theta_{\\text{GS}}'],
        'descriptions': '4 modular, 1 gross substitutes'
    },
    'QuadSupermodularNetwork': {
        'names': ['\\theta_{\\text{MOD}}'] * 4 + ['\\theta_{\\text{QUAD}}'],
        'descriptions': '4 modular, 1 quadratic'
    },
    'LinearKnapsack': {
        'names': ['\\theta_{\\text{MOD}}'],
        'descriptions': 'modular features'
    },
    'QuadKnapsack': {
        'names': ['\\theta_{\\text{MOD}}'] * 5 + ['\\theta_{\\text{QUAD}}'],
        'descriptions': '5 modular, 1 quadratic'
    },
    'PlainSingleItem': {
        'names': ['\\theta_{\\text{MOD}}'] * 5,
        'descriptions': '5 modular agent features'
    }
}

# Method display names
METHOD_NAMES = {
    'row_generation': 'Row Gen',
    'row_generation_1slack': 'Row Gen (1-Slack)',
    'ellipsoid': 'Ellipsoid'
}

# Experiment type titles
EXPERIMENT_TITLES = {
    'Greedy': 'Gross Substitutes',
    'QuadSupermodularNetwork': 'Supermodular Setting',
    'LinearKnapsack': 'Linear Knapsack Setting',
    'QuadKnapsack': 'Quadratic Knapsack Setting',
    'PlainSingleItem': 'Plain Single Item Setting'
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
    theta_true_col = f'theta_true_{param_idx}'
    theta_est_col = f'theta_{param_idx}'
    
    if theta_true_col not in df_group.columns or theta_est_col not in df_group.columns:
        return np.nan, np.nan
    
    # Convert to numeric, handling any string values
    theta_true = pd.to_numeric(df_group[theta_true_col], errors='coerce').values
    theta_est = pd.to_numeric(df_group[theta_est_col], errors='coerce').values
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(theta_true) | np.isnan(theta_est))
    if valid_mask.sum() == 0:
        return np.nan, np.nan
    
    theta_true = theta_true[valid_mask]
    theta_est = theta_est[valid_mask]
    
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
            rmse, bias = compute_aggregate_rmse_bias(df_method, k)
            rmse_list.append(rmse)
            bias_list.append(bias)
        
        # Aggregate: use mean RMSE and mean absolute bias (or just mean bias)
        agg_rmse = np.mean(rmse_list)
        agg_bias = np.mean(bias_list)
        aggregated[param_type] = (agg_rmse, agg_bias)
    
    return aggregated


def generate_table_for_size(df_size, methods, param_config, num_agents, num_items):
    """Generate LaTeX table rows for a specific problem size."""
    param_names = param_config['names']
    unique_params = sorted(set(param_names), key=lambda x: param_names.index(x))
    num_unique = len(unique_params)
    
    lines = []
    
    for method in methods:
        df_method = df_size[df_size['method'] == method]
        if len(df_method) == 0:
            continue
        
        runtime = df_method['time_s'].mean()
        method_name = METHOD_NAMES.get(method, method)
        
        # Aggregate parameters by type
        aggregated = aggregate_parameters_by_type(df_method, param_config)
        
        # Get RMSE and Bias for each unique parameter type
        rmse_values = [format_number(aggregated[p][0]) for p in unique_params]
        bias_values = [format_number(aggregated[p][1]) for p in unique_params]
        
        # Format row
        if num_unique == 1:
            # Simple format for single parameter type
            row = f"{method_name:25s} & {format_number(runtime):>10s} & {rmse_values[0]:>10s} & {bias_values[0]:>10s} \\\\"
        else:
            # Multi-parameter format
            rmse_str = ' & '.join(rmse_values)
            bias_str = ' & '.join(bias_values)
            row = f"{method_name:25s} & {format_number(runtime):>10s} & {rmse_str} & {bias_str} \\\\"
        
        lines.append(row)
    
    return lines, unique_params


def generate_latex_table(experiment_type, results_data, output_file):
    """
    Generate LaTeX tables in 3x3 grid format for an experiment type.
    
    Args:
        experiment_type: Type of experiment (Greedy, etc.)
        results_data: Dict mapping (num_agents, num_items) -> DataFrame
        output_file: File to write LaTeX to
    """
    param_config = PARAM_NAMES.get(experiment_type, PARAM_NAMES['LinearKnapsack'])
    title = EXPERIMENT_TITLES.get(experiment_type, experiment_type)
    param_names = param_config['names']
    unique_params = sorted(set(param_names), key=lambda x: param_names.index(x))
    
    # Organize sizes dynamically based on actual data
    # Extract unique agent and item counts from actual results
    agent_counts = sorted(set(n for n, m in results_data.keys()))
    item_counts = sorted(set(m for n, m in results_data.keys()))
    
    # Use all available sizes, don't force 3x3 grid
    # This makes the table flexible to show whatever data exists
    
    lines = []
    lines.append(f"\\begin{{table}}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    # Dynamic grid description
    grid_desc = f"{len(agent_counts)}$\\times${len(item_counts)} Grid" if len(agent_counts) > 1 or len(item_counts) > 1 else "Single Size"
    lines.append(f"\\caption{{Numerical Experiment: {title} ({grid_desc}: $N \\times M$)}}")
    lines.append(f"\\label{{tab:{experiment_type.lower()}}}")
    lines.append("\\begin{threeparttable}")
    # Dynamic column specification based on actual data
    num_cols = 1 + len(agent_counts) * len(item_counts)
    col_spec = "l " + " ".join(["c"] * (num_cols - 1))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header: M (Items) across top
    if len(item_counts) > 0:
        header1 = f"Method & \\multicolumn{{{num_cols - 1}}}{{c}}{{$M$ (Items)}} \\\\"
        lines.append(header1)
        lines.append(f"\\cmidrule(lr){{2-{num_cols}}}")
        
        # Header row 2: Item values (grouped by item count)
        if len(item_counts) > 1:
            header2_parts = []
            for m in item_counts:
                header2_parts.append(f"\\multicolumn{{{len(agent_counts)}}}{{c}}{{{m}}}")
            header2 = " & " + " & ".join(header2_parts) + " \\\\"
            lines.append(header2)
            # Add cmidrules for each item group
            cmidrules = []
            for i in range(len(item_counts)):
                start = 2 + i * len(agent_counts)
                end = start + len(agent_counts) - 1
                cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
            lines.append(" ".join(cmidrules))
        else:
            # Single item count, no grouping needed
            lines.append(f" & \\multicolumn{{{len(agent_counts)}}}{{c}}{{{item_counts[0]}}} \\\\")
            lines.append(f"\\cmidrule(lr){{2-{num_cols}}}")
        
        # Header row 3: Agent values (repeated for each item count)
        header3_parts = []
        for m in item_counts:
            for n in agent_counts:
                header3_parts.append(f"$N={n}$")
        header3 = " & " + " & ".join(header3_parts) + " \\\\"
        lines.append(header3)
    lines.append("\\midrule")
    
    # Get all methods
    all_methods = set()
    for df_size in results_data.values():
        all_methods.update(df_size['method'].unique())
    methods = sorted([m for m in ['row_generation', 'row_generation_1slack'] if m in all_methods])
    
    # Helper function to extract parameter label from LaTeX string
    def extract_param_label(param_latex):
        """Extract parameter label like 'MOD', 'GS', 'QUAD' from LaTeX string."""
        # Handle LaTeX strings like \theta_{\text{MOD}}, \theta_{\text{GS}}, etc.
        if 'MOD' in param_latex:
            return 'MOD'
        elif 'GS' in param_latex:
            return 'GS'
        elif 'QUAD' in param_latex:
            return 'QUAD'
        # Fallback: try to extract from structure
        s = param_latex.replace('\\', '').replace('{', '').replace('}', '').replace('theta', '').replace('text', '').replace('_', '').strip()
        return s if s else 'MOD'  # Default fallback
    
    # Helper function to get value for a (N, M) combination
    def get_value(method, n, m, metric_type='runtime'):
        """Get value for a specific method, size, and metric."""
        key = (n, m)
        if key not in results_data:
            return "---"
        
        df_size = results_data[key]
        df_method = df_size[df_size['method'] == method]
        
        if len(df_method) == 0:
            return "---"
        
        if metric_type == 'runtime':
            return format_number(df_method['time_s'].mean())
        else:
            aggregated = aggregate_parameters_by_type(df_method, param_config)
            # metric_type is like 'rmse_MOD' or 'bias_GS'
            parts = metric_type.split('_', 1)
            if len(parts) != 2:
                return "---"
            metric, param_label = parts
            # Find matching param type in aggregated (match by extracted label)
            for param_type in aggregated.keys():
                if extract_param_label(param_type) == param_label:
                    idx = 0 if metric == 'rmse' else 1
                    return format_number(aggregated[param_type][idx])
            return "---"
    
    # Generate table: one section per metric type
    # 1. Runtime
    lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{Runtime (s)}}}} \\\\")
    for method in methods:
        row_parts = [METHOD_NAMES.get(method, method)]
        for m in item_counts:
            for n in agent_counts:
                row_parts.append(get_value(method, n, m, 'runtime'))
        lines.append(" & ".join(row_parts) + " \\\\")
    lines.append(f"\\cmidrule(lr){{1-{num_cols}}}")
    
    # 2. RMSE for each parameter type
    for param_type in unique_params:
        param_label = extract_param_label(param_type)
        lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{RMSE}} ${param_type}$}} \\\\")
        for method in methods:
            row_parts = [METHOD_NAMES.get(method, method)]
            for m in item_counts:
                for n in agent_counts:
                    row_parts.append(get_value(method, n, m, f'rmse_{param_label}'))
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append(f"\\cmidrule(lr){{1-{num_cols}}}")
    
    # 3. Bias for each parameter type
    for param_type in unique_params:
        param_label = extract_param_label(param_type)
        lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textit{{Bias}} ${param_type}$}} \\\\")
        for method in methods:
            row_parts = [METHOD_NAMES.get(method, method)]
            for m in item_counts:
                for n in agent_counts:
                    row_parts.append(get_value(method, n, m, f'bias_{param_label}'))
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append(f"\\cmidrule(lr){{1-{num_cols}}}")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\footnotesize")
    available_sizes = sorted(results_data.keys())
    size_str = ", ".join([f"({n},{m})" for n, m in available_sizes])
    lines.append(f"\\item Runtime in seconds. Each cell shows $(N,M)$ combination. Data available for $(N,M) \\in \\{{{size_str}\\}}$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    lines.append("")
    
    return lines


def load_results_from_directory(directory):
    """Load all results CSV files from a directory."""
    results_data = defaultdict(dict)
    
    # Try both patterns: subdirectories with results.csv, and also check for results_raw.csv in the directory itself
    csv_files = list(Path(directory).glob("*/results.csv"))
    # Also check if there's a results_raw.csv in the directory (for aggregated results)
    results_raw = Path(directory) / "results_raw.csv"
    if results_raw.exists():
        csv_files.append(results_raw)
    
    for csv_file in csv_files:
        try:
            # Read CSV with error handling for malformed rows
            df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
            df = df[df['method'] != 'ERROR'].copy()
            
            if len(df) == 0:
                continue
            
            # Check if theta columns exist
            # Convert to numeric, handling any string values
            try:
                num_features = int(pd.to_numeric(df['num_features'].iloc[0], errors='coerce'))
            except (ValueError, TypeError):
                print(f"Skipping {csv_file}: invalid num_features value")
                continue
            if pd.isna(num_features):
                print(f"Skipping {csv_file}: num_features is NaN")
                continue
            has_theta_cols = all(f'theta_{k}' in df.columns for k in range(num_features))
            has_theta_true_cols = all(f'theta_true_{k}' in df.columns for k in range(num_features))
            
            if not (has_theta_cols and has_theta_true_cols):
                print(f"Skipping {csv_file}: missing theta columns (may be old format)")
                continue
            
            # Get metadata and group by each unique size in the CSV
            experiment_type = df['subproblem'].iloc[0]
            
            # Group by each unique size in the CSV (since CSV can contain multiple sizes)
            for (num_agents, num_items), size_df in df.groupby(['num_agents', 'num_items']):
                key = (int(num_agents), int(num_items))
                if key not in results_data[experiment_type]:
                    results_data[experiment_type][key] = []
                results_data[experiment_type][key].append(size_df)
            
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    # Combine DataFrames for each size
    combined_data = {}
    for exp_type, sizes in results_data.items():
        combined_data[exp_type] = {}
        for size_key, df_list in sizes.items():
            combined_data[exp_type][size_key] = pd.concat(df_list, ignore_index=True)
    
    return combined_data


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from experiment results')
    parser.add_argument('results_dir', type=str, nargs='?', default='experiments_paper',
                       help='Directory containing experiment results (default: experiments_paper)')
    parser.add_argument('--output', '-o', type=str, default='experiments_paper/tables.tex',
                       help='Output LaTeX file (default: experiments_paper/tables.tex)')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Specific experiment type to generate (default: all)')
    
    args = parser.parse_args()
    
    # Load results
    results_data = load_results_from_directory(args.results_dir)
    
    if not results_data:
        print(f"Error: No valid results found in {args.results_dir}")
        return 1
    
    # Generate LaTeX
    all_lines = []
    all_lines.append("% Generated LaTeX tables from experiment results")
    all_lines.append("% Use with beamer document class")
    all_lines.append("")
    all_lines.append("\\section{Numerical Experiments}")
    all_lines.append("")
    
    # Determine which experiments to generate
    # Map lowercase experiment names to actual keys
    exp_name_map = {
        'greedy': 'Greedy',
        'supermod': 'QuadSupermodularNetwork',
        'knapsack': 'LinearKnapsack',
        'quadknapsack': 'QuadKnapsack',
        'plain_single_item': 'PlainSingleItem'
    }
    
    if args.experiment:
        # Try to map the argument to the actual key
        exp_lower = args.experiment.lower()
        if exp_lower in exp_name_map:
            experiments_to_generate = [exp_name_map[exp_lower]]
        elif args.experiment in results_data:
            experiments_to_generate = [args.experiment]
        else:
            print(f"Error: Unknown experiment type: {args.experiment}")
            return 1
    else:
        experiments_to_generate = sorted(results_data.keys())
    
    for exp_type in experiments_to_generate:
        if exp_type not in results_data:
            print(f"Warning: No results found for {exp_type}")
            continue
        
        exp_results = results_data[exp_type]
        lines = generate_latex_table(exp_type, exp_results, args.output)
        all_lines.extend(lines)
        all_lines.append("")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_lines))
    
    print(f"LaTeX tables generated: {output_path}")
    print(f"Experiments included: {', '.join(experiments_to_generate)}")
    
    return 0


if __name__ == '__main__':
    exit(main())
