#!/bin/bash
# Comprehensive performance comparison script
# Tests main vs feature branch at multiple scales and MPI ranks

set -e

PROJECT_ROOT="/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation"
cd "$PROJECT_ROOT"

# Activate virtual environment
source .bundle/bin/activate

# Test configurations
SCALES=("128x100" "256x150" "512x200")
RANKS=(4 8 12)

# Parse scale into agents and items
parse_scale() {
    local scale=$1
    case $scale in
        "128x100")
            echo "128 100"
            ;;
        "256x150")
            echo "256 150"
            ;;
        "512x200")
            echo "512 200"
            ;;
    esac
}

# Run test on a branch
run_test() {
    local branch=$1
    local scale=$1
    local ranks=$2
    local output_file=$3
    
    local agents items
    read agents items <<< $(parse_scale $scale)
    
    echo "Testing $branch branch: $scale with $ranks ranks..."
    
    mpirun -n $ranks python experiments/mpi_performance_test/run_single_test.py \
        --agents $agents \
        --items $items \
        --simuls 5 \
        --features 6 \
        --max-iters 50 \
        --seed 42 \
        > "$output_file" 2>&1
    
    # Extract key metrics
    total_time=$(grep "Total time:" "$output_file" | awk '{print $3}' | sed 's/s//')
    iterations=$(grep "Total iterations:" "$output_file" | awk '{print $3}')
    pricing=$(grep "pricing:" "$output_file" | head -1 | awk '{print $2}' | sed 's/s//')
    mpi_gather=$(grep "mpi_gather:" "$output_file" | head -1 | awk '{print $2}' | sed 's/s//')
    compute_features=$(grep "compute_features:" "$output_file" | head -1 | awk '{print $2}' | sed 's/s//')
    gather_features=$(grep "gather_features:" "$output_file" | head -1 | awk '{print $2}' | sed 's/s//')
    
    echo "$branch,$scale,$ranks,$total_time,$iterations,$pricing,$mpi_gather,$compute_features,$gather_features"
}

# Create results directory
RESULTS_DIR="experiments/mpi_performance_test/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting comprehensive performance tests..."
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Test main branch
echo "Testing MAIN branch..."
git checkout main
git pull origin main

for scale in "${SCALES[@]}"; do
    for ranks in "${RANKS[@]}"; do
        output_file="$RESULTS_DIR/main_${scale}_${ranks}ranks.out"
        run_test "main" "$scale" "$ranks" "$output_file" >> "$RESULTS_DIR/main_results.csv"
        sleep 2  # Brief pause between tests
    done
done

# Test feature branch
echo ""
echo "Testing FEATURE branch..."
git checkout feature/mpi-gather-optimization
git pull origin feature/mpi-gather-optimization

for scale in "${SCALES[@]}"; do
    for ranks in "${RANKS[@]}"; do
        output_file="$RESULTS_DIR/feature_${scale}_${ranks}ranks.out"
        run_test "feature" "$scale" "$ranks" "$output_file" >> "$RESULTS_DIR/feature_results.csv"
        sleep 2  # Brief pause between tests
    done
done

# Generate comparison report
echo ""
echo "Generating comparison report..."
python3 << 'PYTHON_SCRIPT'
import csv
import sys

# Read results
main_results = {}
feature_results = {}

try:
    with open('experiments/mpi_performance_test/results_*/main_results.csv') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                key = f"{parts[1]}_{parts[2]}"
                main_results[key] = parts
except:
    pass

try:
    with open('experiments/mpi_performance_test/results_*/feature_results.csv') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                key = f"{parts[1]}_{parts[2]}"
                feature_results[key] = parts
except:
    pass

# Print comparison
print("\n" + "="*80)
print("PERFORMANCE COMPARISON: MAIN vs FEATURE BRANCH")
print("="*80)
print(f"{'Scale':<12} {'Ranks':<6} {'Main (s)':<10} {'Feature (s)':<12} {'Diff %':<10} {'Winner':<10}")
print("-"*80)

for key in sorted(set(list(main_results.keys()) + list(feature_results.keys()))):
    if key in main_results and key in feature_results:
        main = main_results[key]
        feature = feature_results[key]
        
        scale = main[1]
        ranks = main[2]
        main_time = float(main[3]) if main[3] else 0
        feature_time = float(feature[3]) if feature[3] else 0
        
        if main_time > 0:
            diff_pct = ((feature_time - main_time) / main_time) * 100
            winner = "FEATURE" if feature_time < main_time else "MAIN" if main_time < feature_time else "TIE"
            print(f"{scale:<12} {ranks:<6} {main_time:<10.3f} {feature_time:<12.3f} {diff_pct:+9.1f}% {winner:<10}")

PYTHON_SCRIPT

echo ""
echo "Tests complete! Results saved to: $RESULTS_DIR"



