#!/usr/bin/env python3
"""Fix CSV files by removing rows with mismatched columns."""
import csv
import sys
import argparse
from pathlib import Path

def fix_csv(csv_path, quiet=False):
    """Fix CSV file by removing/fixing rows with mismatched columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        if not quiet:
        print(f"File not found: {csv_path}")
        return False
    
    # Read and check
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)
        
        rows = [header]
        fixed = 0
        skipped = 0
        
        for row in reader:
            if len(row) == expected_cols:
                rows.append(row)
            else:
                skipped += 1
                # Try to fix by truncating or padding
                if len(row) > expected_cols:
                    rows.append(row[:expected_cols])
                    fixed += 1
                elif len(row) < expected_cols:
                    rows.append(row + [''] * (expected_cols - len(row)))
                    fixed += 1
        
    # Only write if there were issues
    if fixed == 0 and skipped == 0:
        return False  # No fixes needed
    
    if not quiet:
        print(f"{csv_path.name}: Expected {expected_cols} cols, fixed {fixed} rows, skipped {skipped} rows")
    
    # Write back
    backup = csv_path.with_suffix('.csv.backup2')
    csv_path.rename(backup)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    if not quiet:
    print(f"  → Fixed and saved, backup: {backup.name}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix CSV files with mismatched columns')
    parser.add_argument('csv_file', type=str, nargs='?', default=None,
                       help='CSV file to fix (default: fix all experiment results.csv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output (useful for automated runs)')
    
    args = parser.parse_args()
    
    if args.csv_file:
        # Fix specific file
        fix_csv(args.csv_file, quiet=args.quiet)
    else:
        # Fix all experiment results
    for exp in ['greedy', 'supermod', 'knapsack', 'supermodknapsack']:
        csv_path = Path(exp) / 'results.csv'
        if csv_path.exists():
                fix_csv(csv_path, quiet=args.quiet)
