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
        'names': ['\\theta_{\\text{MOD}}'] * 5 + ['\\theta_{\\text{QUAD}}'],
        'descriptions': '5 modular, 1 quadratic'
    },
    'LinearKnapsack': {
        'names': ['\\theta_{\\text{MOD}}'],
        'descriptions': 'modular features'
    },
    'QuadKnapsack': {
        'names': ['\\theta_{\\text{MOD}}'] * 5 + ['\\theta_{\\text{QUAD}}'],
        'descriptions': '5 modular, 1 quadratic'
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
    'QuadKnapsack': 'Quadratic Knapsack Setting'
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
    Generate LaTeX tables for an experiment type.
    
    Args:
        experiment_type: Type of experiment (Greedy, etc.)
        results_data: Dict mapping (num_agents, num_items) -> DataFrame
        output_file: File to write LaTeX to
    """
    param_config = PARAM_NAMES.get(experiment_type, PARAM_NAMES['LinearKnapsack'])
    title = EXPERIMENT_TITLES.get(experiment_type, experiment_type)
    param_names = param_config['names']
    unique_params = sorted(set(param_names), key=lambda x: param_names.index(x))
    num_unique = len(unique_params)
    
    # Sort by num_agents
    sorted_sizes = sorted(results_data.keys(), key=lambda x: x[0])
    
    lines = []
    lines.append(f"\\begin{{frame}}{{Numerical Experiment: {title}}}")
    lines.append("")
    
    # Slide 1: Description
    lines.append("\\only<1>{")
    lines.append("")
    lines.append("\\begin{itemize}")
    
    # Add description based on experiment type
    if experiment_type == 'Greedy':
        lines.append("    \\item $J = 100$ items, $K = 5$ features: 4 modular, 1 gross substitutes.")
        lines.append("    \\item Agent $i$'s utility for bundle $B$:")
        lines.append("    \\[")
        lines.append("    \\sum_{j \\in B} \\phi_{ij}^\\top \\theta_{\\text{MOD}} -|B|^2\\theta_{\\text{GS}} + \\varepsilon_{iB}.")
        lines.append("    \\]")
        lines.append("    \\item Modular component $\\phi_{ijk}$ drawn i.i.d. from half-normal distribution.")
        lines.append("    \\item Modular errors: $\\varepsilon_{iB} = \\sum_{j \\in B} \\nu_{ij},\\; \\nu_{ij} \\sim \\mathcal{N}(0, \\sigma^2).$")
    elif experiment_type == 'QuadSupermodularNetwork':
        lines.append("    \\item $J = 100$ items, $K = 6$ features: 5 modular, 1 quadratic.")
        lines.append("    \\item Agent $i$'s utility for bundle $B$:")
        lines.append("    \\[")
        lines.append("    -\\sum_{j \\in B} \\phi_{ij}^\\top \\theta_{\\text{MOD}} + \\sum_{\\substack{j < j' \\\\ j,j' \\in B}} \\phi_{jj'}^\\top \\theta_{\\text{QUAD}} + \\varepsilon_{iB}.")
        lines.append("    \\]")
        lines.append("    \\item Modular component $\\phi_{ijk}$ drawn i.i.d. from half-normal distribution.")
        lines.append("    \\item Quadratic component $\\phi_{jj'}$ is i.i.d.\\ Bernoulli$(0.2)$.")
        lines.append("    \\item Modular errors: $\\varepsilon_{iB} = \\sum_{j \\in B} \\nu_{ij},\\; \\nu_{ij} \\sim \\mathcal{N}(0, \\sigma^2).$")
    elif experiment_type == 'LinearKnapsack':
        lines.append("    \\item $J = 100$ items, $K = 5$ modular features.")
        lines.append("    \\item Agent $i$'s utility for bundle $B$:")
        lines.append("    \\[")
        lines.append("    U_{{iB}} = \\sum_{{j \\in B}} \\phi_{{ij}}^\\top \\theta + \\varepsilon_{{iB}}.")
        lines.append("    \\]")
        lines.append("    \\item Choice set:")
        lines.append("    \\[")
        lines.append("    \\mathcal{{B}}_i = \\{{ B \\subseteq [J] \\;|\\; \\sum_{{j \\in B}} w_j \\leq W_i \\}}.")
        lines.append("    \\]")
        lines.append("    \\item Modular features $\\phi_{{ijk}}$ drawn from half-normal distribution.")
        lines.append("    \\item Modular errors: $\\varepsilon_{{iB}} = \\sum_{{j \\in B}} \\nu_{{ij}},\\; \\nu_{{ij}} \\sim \\mathcal{{N}}(0, \\sigma^2).$")
        lines.append("    \\item Item weights $w_j \\sim_{{i.i.d.}} \\text{{Uniform}}\\{{1,10\\}}$ and $W_i$ drawn around $\\tfrac{{1}}{{2}} \\sum_{{j \\in [J]}} w_j.$")
    elif experiment_type == 'QuadKnapsack':
        lines.append("    \\item $J = 100$ items, $K = 6$ features: 5 modular, 1 quadratic.")
        lines.append("    \\item Agent $i$'s utility for bundle $B$ with knapsack constraint:")
        lines.append("    \\[")
        lines.append("    U_{{iB}} = \\sum_{{j \\in B}} \\phi_{{ij}}^\\top \\theta_{\\text{{MOD}}} + \\sum_{\\substack{{j < j' \\\\ j,j' \\in B}}}} \\phi_{{jj'}}^\\top \\theta_{\\text{{QUAD}}} + \\varepsilon_{{iB}}.")
        lines.append("    \\]")
        lines.append("    \\item Choice set: $\\mathcal{{B}}_i = \\{{ B \\subseteq [J] \\;|\\; \\sum_{{j \\in B}} w_j \\leq W_i \\}}$.")
        lines.append("    \\item Modular and quadratic components drawn from half-normal and Bernoulli$(0.2)$ respectively.")
        lines.append("    \\item Modular errors: $\\varepsilon_{{iB}} = \\sum_{{j \\in B}} \\nu_{{ij}},\\; \\nu_{{ij}} \\sim \\mathcal{{N}}(0, \\sigma^2).$")
    
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("}")
    lines.append("")
    
    # Slide 2: Tables for each size
    lines.append("\\only<2>{")
    lines.append("")
    
    # Get all methods present across all sizes
    all_methods = set()
    for df_size in results_data.values():
        all_methods.update(df_size['method'].unique())
    methods = sorted([m for m in ['row_generation', 'row_generation_1slack', 'ellipsoid'] if m in all_methods])
    
    # Determine number of columns needed per size
    num_cols_per_size = 1 + (2 * num_unique)  # Runtime + RMSE columns + Bias columns
    total_cols = 1 + len(sorted_sizes) * num_cols_per_size
    
    # Use landscape if table is very wide (more than 30 columns)
    use_landscape = total_cols > 30
    
    # Start table - all sizes in one table
    if use_landscape:
        lines.append("\\begin{landscape}")
        lines.append("\\begin{table}[htbp]")
    else:
        lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{threeparttable}")
    
    # Build column specification
    colspec = "l "  # Method column
    for _ in sorted_sizes:
        if num_unique == 1:
            colspec += "r r r "  # Runtime, RMSE, Bias
        else:
            colspec += "r " + "r" * num_unique + " " + "r" * num_unique + " "  # Runtime + RMSE params + Bias params
    
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    
    # Header row 1: Size labels (show both I and J)
    header1 = "Method"
    for num_agents, num_items in sorted_sizes:
        if num_unique == 1:
            header1 += f" & \\multicolumn{{3}}{{c}}{{$(I,J)=({num_agents},{num_items})$}}"
        else:
            header1 += f" & \\multicolumn{{{num_cols_per_size}}}{{c}}{{$(I,J)=({num_agents},{num_items})$}}"
    header1 += " \\\\"
    lines.append(header1)
    
    # Header row 2: Column types
    header2 = " & "
    for num_agents, num_items in sorted_sizes:
        if num_unique == 1:
            header2 += "Runtime (s) & RMSE & Bias & "
        else:
            header2 += "Runtime (s) & \\multicolumn{" + str(num_unique) + "}{c}{RMSE} & \\multicolumn{" + str(num_unique) + "}{c}{Bias} & "
    header2 = header2.rstrip(" & ") + " \\\\"
    lines.append(header2)
    
    # Header row 3: Parameter names (only for multi-param case)
    if num_unique > 1:
        header3 = " & "
        for num_agents, num_items in sorted_sizes:
            param_cols = " & ".join([f"${p}$" for p in unique_params])
            header3 += f" & {param_cols} & {param_cols} & "
        header3 = header3.rstrip(" & ") + " \\\\"
        lines.append(header3)
        
        # Cmidrules
        cmidrule = "\\cmidrule(lr){2-" + str(1+num_cols_per_size) + "}"
        for i, _ in enumerate(sorted_sizes[1:], 1):
            start = 2 + i * num_cols_per_size
            end = start + num_cols_per_size - 1
            cmidrule += f" \\cmidrule(lr){{{start}-{end}}}"
        lines.append(cmidrule)
    else:
        # Cmidrules for single param
        cmidrule = "\\cmidrule(lr){2-4}"
        for i in range(1, len(sorted_sizes)):
            start = 2 + i * 3
            cmidrule += f" \\cmidrule(lr){{{start}-{start+2}}}"
        lines.append(cmidrule)
    
    lines.append("\\midrule")
    
    # Generate rows for each method
    for method in methods:
        row_parts = [METHOD_NAMES.get(method, method)]
        
        for num_agents, num_items in sorted_sizes:
            df_size = results_data[(num_agents, num_items)]
            df_method = df_size[df_size['method'] == method]
            
            if len(df_method) == 0:
                # No data for this method/size
                if num_unique == 1:
                    row_parts.extend(["---", "---", "---"])
                else:
                    row_parts.extend(["---"] + ["---"] * num_unique + ["---"] * num_unique)
                continue
            
            runtime = df_method['time_s'].mean()
            aggregated = aggregate_parameters_by_type(df_method, param_config)
            
            # Add runtime
            row_parts.append(format_number(runtime))
            
            # Add RMSE and Bias for each unique parameter type
            for param_type in unique_params:
                row_parts.append(format_number(aggregated[param_type][0]))  # RMSE
            for param_type in unique_params:
                row_parts.append(format_number(aggregated[param_type][1]))  # Bias
        
        row = " & ".join(row_parts) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    if use_landscape:
        lines.append("\\end{landscape}")
    lines.append("")
    
    lines.append("}")
    lines.append("\\end{frame}")
    lines.append("")
    
    return lines


def load_results_from_directory(directory):
    """Load all results CSV files from a directory."""
    results_data = defaultdict(dict)
    
    for csv_file in Path(directory).glob("*/results.csv"):
        try:
            df = pd.read_csv(csv_file)
            df = df[df['method'] != 'ERROR'].copy()
            
            if len(df) == 0:
                continue
            
            # Check if theta columns exist
            num_features = int(df['num_features'].iloc[0])
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
    experiments_to_generate = [args.experiment] if args.experiment else sorted(results_data.keys())
    
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
