#!/bin/env python
"""
Generate LaTeX table with parameter estimates and standard errors.

Reads from theta_hat.csv and se_non_fe.csv, fills table for both δ=2 and δ=4.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import numpy as np

# Non-FE parameter indices and names
NON_FE_INDICES = [0, 494, 495, 496]
PARAM_CSV_NAMES = ["bidder_elig_pop", "pop_distance", "travel_survey", "air_travel"]

# Parameter names for LaTeX table
PARAMETER_NAMES = [
    "Bidder eligibility $\\times$ population",
    "Population/distance",
    "Trips between markets in a package",
    "Total trips between airports in markets"
]

PARAMETER_DESCRIPTIONS = [
    "",  # Bidder eligibility × population (no description needed)
    "two markets in a package",
    "in the American Travel Survey",
    "in a package (thousands)"
]


def load_theta_from_csv(delta):
    """Load theta estimates for given delta from theta_hat.csv."""
    csv_path = os.path.join(BASE_DIR, "..", "estimation_results", "theta_hat.csv")
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find rows with matching delta
    matching = [r for r in rows if str(r.get("delta", "")) == str(delta)]
    if not matching:
        return None
    
    # Get the most recent row (last one)
    row = matching[-1]
    
    # Extract theta values for non-FE parameters
    theta_values = []
    for idx in NON_FE_INDICES:
        key = f"theta_{idx}"
        if key in row and row[key]:
            theta_values.append(float(row[key]))
        else:
            theta_values.append(np.nan)
    
    return np.array(theta_values)


def load_se_from_csv(delta):
    """Load standard errors for given delta from se_non_fe.csv."""
    csv_path = os.path.join(BASE_DIR, "..", "estimation_results", "se_non_fe.csv")
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find rows with matching delta
    matching = [r for r in rows if str(r.get("delta", "")) == str(delta)]
    if not matching:
        return None
    
    # Get the most recent row (last one)
    row = matching[-1]
    
    # Extract SE values
    se_values = []
    for name in PARAM_CSV_NAMES:
        key = f"se_{name}"
        if key in row and row[key]:
            se_values.append(float(row[key]))
        else:
            se_values.append(np.nan)
    
    return np.array(se_values)


def load_results_for_delta(delta):
    """Load theta and SE for a given delta."""
    theta = load_theta_from_csv(delta)
    se = load_se_from_csv(delta)
    
    if theta is None:
        theta = np.full(4, np.nan)
    if se is None:
        se = np.full(4, np.nan)
    
    return theta, se


def format_value(val, decimals=2):
    """Format a value for LaTeX, return '---' if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def generate_latex_table(results_delta2, results_delta4, output_path=None):
    """
    Generate LaTeX table with estimates for both δ=2 and δ=4.
    
    Args:
        results_delta2: (theta, se) tuple for delta=2
        results_delta4: (theta, se) tuple for delta=4
        output_path: Path to save the LaTeX file (optional)
    """
    theta2, se2 = results_delta2
    theta4, se4 = results_delta4
    
    lines = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("& \\multicolumn{2}{c}{$\\delta = 4$} & \\multicolumn{2}{c}{$\\delta = 2$} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append("\\textbf{} & \\textbf{Coef.} & \\textbf{SE} & \\textbf{Coef.} & \\textbf{SE} \\\\")
    lines.append("\\hline")
    
    for i, (param_name, param_desc) in enumerate(zip(PARAMETER_NAMES, PARAMETER_DESCRIPTIONS)):
        # Format values
        coef4 = format_value(theta4[i])
        se4_str = format_value(se4[i])
        coef2 = format_value(theta2[i])
        se2_str = format_value(se2[i])
        
        # Main row with parameter name
        lines.append(f"{param_name} & {coef4} & {se4_str} & {coef2} & {se2_str} \\\\")
        
        # Description row (if any)
        if param_desc:
            lines.append(f"\\quad \\textit{{{param_desc}}} & & & & \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    
    table_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table_str)
        print(f"Table saved to: {output_path}")
    
    return table_str


def main():
    """Main function to generate the table."""
    print("=" * 60)
    print("GENERATING LATEX TABLE FOR PAPER")
    print("=" * 60)
    
    # Load results for both deltas
    print("\nLoading results for δ=4...")
    theta4, se4 = load_results_for_delta(4)
    print(f"  Theta: {['%.2f' % t if not np.isnan(t) else 'N/A' for t in theta4]}")
    print(f"  SE: {['%.2f' % s if not np.isnan(s) else 'N/A' for s in se4]}")
    
    print("\nLoading results for δ=2...")
    theta2, se2 = load_results_for_delta(2)
    print(f"  Theta: {['%.2f' % t if not np.isnan(t) else 'N/A' for t in theta2]}")
    print(f"  SE: {['%.2f' % s if not np.isnan(s) else 'N/A' for s in se2]}")
    
    # Generate table
    output_path = os.path.join(BASE_DIR, "parameter_estimates.tex")
    table_str = generate_latex_table(
        results_delta2=(theta2, se2),
        results_delta4=(theta4, se4),
        output_path=output_path
    )
    
    print("\n" + "=" * 60)
    print("LATEX TABLE:")
    print("=" * 60)
    print(table_str)
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
