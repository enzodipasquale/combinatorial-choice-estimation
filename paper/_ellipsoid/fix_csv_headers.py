#!/usr/bin/env python3
"""
Fix CSV headers to include missing timing columns.
This script recreates the CSV headers with correct column structure.
"""
import os
import csv
import yaml

def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def fix_csv_header(exp_dir: str):
    """Fix CSV header to include all required columns."""
    config_path = os.path.join(exp_dir, 'config.yaml')
    results_path = os.path.join(exp_dir, 'results.csv')
    
    if not os.path.exists(results_path):
        print(f"  {exp_dir}: No results.csv found")
        return
    
    # Load config to get num_features
    cfg = load_yaml_config(config_path)
    num_features = cfg['dimensions']['num_features']
    
    # Read existing data
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        existing_cols = reader.fieldnames
        rows = list(reader)
    
    # Check if timing columns are missing
    expected_timing_cols = ['timing_compute', 'timing_solve', 'timing_comm',
                           'timing_compute_pct', 'timing_solve_pct', 'timing_comm_pct']
    has_timing = all(col in existing_cols for col in expected_timing_cols)
    
    if has_timing:
        print(f"  {exp_dir}: Header already correct ({len(existing_cols)} columns)")
        return
    
    print(f"  {exp_dir}: Fixing header (current: {len(existing_cols)} columns)")
    
    # Build correct column list
    base_cols = ['replication','seed','method','time_s','obj_value',
                 'num_agents','num_items','num_features','num_simuls','sigma','subproblem']
    cols = base_cols.copy()
    cols.extend(expected_timing_cols)
    cols.extend([f'theta_true_{k}' for k in range(num_features)])
    cols.extend([f'theta_{k}' for k in range(num_features)])
    
    # Add error column if it exists in data
    if 'error' in existing_cols:
        cols.append('error')
    
    # Update rows - add missing columns with empty string and remove None keys
    cleaned_rows = []
    for row in rows:
        cleaned_row = {}
        for col in cols:
            if col in row and row[col] is not None:
                cleaned_row[col] = row[col]
            else:
                cleaned_row[col] = ''
        cleaned_rows.append(cleaned_row)
    
    # Write back with correct header
    backup_path = results_path + '.backup'
    os.rename(results_path, backup_path)
    
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"    → Fixed! Backup saved to {backup_path}")
    print(f"    → New header has {len(cols)} columns")

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    experiments = ['greedy', 'supermod', 'knapsack', 'supermodknapsack']
    
    print("Fixing CSV headers...")
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        if os.path.exists(exp_dir):
            fix_csv_header(exp_dir)
        else:
            print(f"  {exp}: Directory not found")
    
    print("\nDone!")

