#!/bin/bash
# Cleanup script for experiments_paper directory
# Removes temporary files, test files, and old job outputs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning up experiments_paper directory..."

# Remove test files
echo "Removing test files..."
rm -f test_*.sbatch test_*.err test_*.out

# Remove backup files
echo "Removing backup files..."
find . -name "*.backup" -o -name "*.backup2" | xargs rm -f 2>/dev/null || true

# Remove test configuration files
echo "Removing test configuration files..."
find . -name "test_*.yaml" -type f -delete
find . -name "test_output_*.log" -type f -delete

# Remove temporary files
echo "Removing temporary files..."
find . -name "results_temp.csv" -type f -delete
find . -name "*.tmp" -type f -delete

# Remove duplicate tables.tex from experiment directories (keep only in 00_RESULTS/)
echo "Removing duplicate tables.tex files..."
for exp in 01_experiments/*; do
    rm -f "$exp/tables.tex" 2>/dev/null || true
done

# Move any stray SLURM output files to 00_LOGS_SLURM/
echo "Organizing SLURM output files..."
mkdir -p 00_LOGS_SLURM
find . -maxdepth 1 -name "*.out" -o -name "*.err" | while read file; do
    if [ -f "$file" ] && [ "$(dirname "$file")" = "." ]; then
        mv "$file" 00_LOGS_SLURM/ 2>/dev/null || true
    fi
done

# Remove old SLURM log files (keep only recent ones from last 30 days)
echo "Removing old SLURM log files (older than 30 days)..."
find 00_LOGS_SLURM -name "*.out" -o -name "*.err" | while read file; do
    if [ -f "$file" ]; then
        # Check if file is older than 30 days
        if [ "$(find "$file" -mtime +30)" ]; then
            rm -f "$file"
        fi
    fi
done

# Remove duplicate root tables.tex if outputs/ has latest
if [ -f "tables.tex" ] && [ -f "tables_all.tex" ]; then
    echo "Removing duplicate root tables.tex (using tables_all.tex instead)..."
    rm -f tables.tex
fi

echo "Cleanup complete!"

