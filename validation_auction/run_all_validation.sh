#!/bin/bash
# Complete validation pipeline script

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "Combinatorial Auction Validation Pipeline"
echo "=========================================="

# Step 1: Extract parameters from real data
echo ""
echo "Step 1: Extracting parameters from real data..."
python extract_parameters.py

# Step 2: Generate synthetic data
echo ""
echo "Step 2: Generating synthetic data..."
python generate_synthetic_data.py --replications 1 --config config_small.yaml

# Step 3: Run validation
echo ""
echo "Step 3: Running validation..."
echo "Submit with: sbatch run_small.sbatch"
echo "Or run directly with:"
echo "  srun ../applications/combinatorial_auction/run-gurobi.bash python run_validation.py --config config_small.yaml"

