#!/bin/bash
# Final extensive comparison: Main vs Feature branch
# Tests: Medium (128x100), Large (256x150), XL (512x200) with 4, 8, 12 ranks

set -e

PROJECT_ROOT="/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation"
cd "$PROJECT_ROOT"
source .bundle/bin/activate

echo "=================================================================================="
echo "EXTENSIVE PERFORMANCE COMPARISON: MAIN vs FEATURE BRANCH"
echo "=================================================================================="
echo "Testing 3 scales × 3 rank counts = 9 configurations per branch"
echo "Total: 18 test runs"
echo "=================================================================================="
echo ""

RESULTS_FILE="experiments/mpi_performance_test/comparison_final_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Test function
test_branch() {
    local branch=$1
    local scale_name=$2
    local agents=$3
    local items=$4
    local ranks=$5
    
    echo "Testing $branch: $scale_name ($agents×$items) with $ranks ranks..."
    
    git checkout $branch > /dev/null 2>&1
    git pull origin $branch > /dev/null 2>&1
    
    output=$(mpirun -n $ranks python experiments/mpi_performance_test/run_single_test.py \
        --agents $agents \
        --items $items \
        --simuls 5 \
        --features 6 \
        --max-iters 50 \
        --seed 42 2>&1)
    
    total_time=$(echo "$output" | grep "Total time:" | tail -1 | awk '{print $3}' | sed 's/s//')
    iterations=$(echo "$output" | grep "Total iterations:" | tail -1 | awk '{print $3}')
    
    echo "  Total time: ${total_time}s, Iterations: $iterations"
    echo "$branch,$scale_name,$ranks,$total_time,$iterations" >> "$RESULTS_FILE"
    
    sleep 1
}

# Initialize results file
echo "branch,scale,ranks,total_time,iterations" > "$RESULTS_FILE"

# Test configurations
declare -a scales=("Medium:128:100" "Large:256:150" "XL:512:200")
declare -a ranks=(4 8 12)

for scale_info in "${scales[@]}"; do
    IFS=':' read -r scale_name agents items <<< "$scale_info"
    
    for num_ranks in "${ranks[@]}"; do
        echo ""
        echo "=================================================================================="
        echo "TESTING: $scale_name ($agents×$items) with $num_ranks ranks"
        echo "=================================================================================="
        
        test_branch "main" "$scale_name" "$agents" "$items" "$num_ranks"
        test_branch "feature/mpi-gather-optimization" "$scale_name" "$agents" "$items" "$num_ranks"
        
        echo ""
    done
done

echo ""
echo "=================================================================================="
echo "COMPARISON SUMMARY"
echo "=================================================================================="
echo ""

# Generate summary
python3 << 'PYTHON'
import csv
import sys

results = {}
with open('experiments/mpi_performance_test/comparison_final_*.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = f"{row['scale']}_{row['ranks']}"
        if key not in results:
            results[key] = {}
        results[key][row['branch']] = {
            'time': float(row['total_time']),
            'iterations': int(row['iterations'])
        }

print(f"{'Scale':<12} {'Ranks':<6} {'Main (s)':<10} {'Feature (s)':<12} {'Diff %':<10} {'Status':<15}")
print("-" * 80)

for key in sorted(results.keys()):
    scale, ranks = key.rsplit('_', 1)
    main = results[key].get('main', {})
    feature = results[key].get('feature/mpi-gather-optimization', {})
    
    if main and feature:
        main_time = main['time']
        feature_time = feature['time']
        
        if main_time > 0:
            diff_pct = ((feature_time - main_time) / main_time) * 100
            
            if abs(diff_pct) < 2:
                status = "SIMILAR"
            elif diff_pct < 0:
                status = f"FASTER {abs(diff_pct):.1f}%"
            else:
                status = f"SLOWER {diff_pct:.1f}%"
            
            print(f"{scale:<12} {ranks:<6} {main_time:<10.3f} {feature_time:<12.3f} {diff_pct:+9.1f}% {status:<15}")

PYTHON

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Test complete!"



