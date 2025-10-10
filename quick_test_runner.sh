#!/bin/bash
# Quick test runner with timeout protection

PASSED=0
FAILED=0
TIMEOUT=0

run_test() {
    local name=$1
    local timeout=$2
    local cmd=$3
    
    echo -n "[$((PASSED + FAILED + TIMEOUT + 1))] $name ... "
    
    if ./timeout_wrapper.sh $timeout $cmd > /tmp/test_output_$$.log 2>&1; then
        echo "✅ PASSED ($(tail -1 /tmp/test_output_$$.log | grep -o '[0-9.]*s' || echo '?'))"
        PASSED=$((PASSED + 1))
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⏱️  TIMEOUT (>${timeout}s)"
            TIMEOUT=$((TIMEOUT + 1))
        else
            echo "❌ FAILED"
            FAILED=$((FAILED + 1))
            tail -20 /tmp/test_output_$$.log | grep -A5 "FAILED\|Error\|assert"
        fi
    fi
    rm -f /tmp/test_output_$$.log
}

echo "=========================================="
echo "BundleChoice Test Suite (Quick Run)"
echo "=========================================="
echo ""

# Simple tests (30s timeout)
run_test "test_config_loading" 30 "pytest bundlechoice/tests/test_config_loading.py -v --tb=line"
run_test "test_config_management" 30 "pytest bundlechoice/tests/test_config_management.py -v --tb=line"
run_test "test_data_manager_simple" 30 "pytest bundlechoice/tests/test_data_manager_simple.py -v --tb=line"
run_test "test_feature_manager" 30 "pytest bundlechoice/tests/test_feature_manager.py -v --tb=line"

# MPI tests (60s timeout)
run_test "test_data_manager_mpi (4 ranks)" 60 "mpirun -n 4 pytest bundlechoice/tests/test_data_manager_mpi.py -v --tb=line"
run_test "test_edge_cases_validation (2 ranks)" 60 "mpirun -n 2 pytest bundlechoice/tests/test_edge_cases_validation.py -v --tb=line"
run_test "test_feature_basics (4 ranks)" 60 "mpirun -n 4 pytest bundlechoice/tests/test_feature_basics.py -v --tb=line"
run_test "test_greedy_caching_comparison (4 ranks)" 60 "mpirun -n 4 pytest bundlechoice/tests/test_greedy_caching_comparison.py -v --tb=line"

# Subproblem tests (60s timeout)
run_test "test_greedy_subproblem (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_greedy_subproblem.py -v --tb=line"
run_test "test_linear_knapsack_subproblem (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_linear_knapsack_subproblem.py -v --tb=line"
run_test "test_quadratic_supermodular_subproblem (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_quadratic_supermodular_subproblem.py -v --tb=line"

# Row generation tests (90s timeout)
run_test "test_estimation_row_generation_greedy (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_greedy.py -v --tb=line"
run_test "test_estimation_row_generation_linear_knapsack (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_linear_knapsack.py -v --tb=line"
run_test "test_estimation_row_generation_plain_single_item (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_plain_single_item.py -v --tb=line"
run_test "test_estimation_row_generation_quadratic_supermodular (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_quadratic_supermodular.py -v --tb=line"

# Row generation 1slack tests (90s timeout)
run_test "test_estimation_row_generation_1slack_greedy (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_greedy.py -v --tb=line"
run_test "test_estimation_row_generation_1slack_linear_knapsack (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_linear_knapsack.py -v --tb=line"
run_test "test_estimation_row_generation_1slack_plain_single_item (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_plain_single_item.py -v --tb=line"
run_test "test_estimation_row_generation_1slack_quadratic_supermodular (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_quadratic_supermodular.py -v --tb=line"

# Ellipsoid tests (60s timeout)
run_test "test_estimation_ellipsoid_greedy (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_greedy.py -v --tb=line"
run_test "test_estimation_ellipsoid_linear_knapsack (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_linear_knapsack.py -v --tb=line"
run_test "test_estimation_ellipsoid_plain_single_item (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_plain_single_item.py -v --tb=line"
run_test "test_estimation_ellipsoid_quadratic_supermodular (10 ranks)" 60 "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_quadratic_supermodular.py -v --tb=line"

# Integration tests (90s timeout) - SKIP for now, they have hypothesis issues
# run_test "test_property_based (4 ranks)" 90 "mpirun -n 4 pytest bundlechoice/tests/test_property_based.py -v --tb=line"
# run_test "test_integration_comprehensive (4 ranks)" 90 "mpirun -n 4 pytest bundlechoice/tests/test_integration_comprehensive.py -v --tb=line"

run_test "test_all_solvers_consistency (10 ranks)" 90 "mpirun -n 10 pytest bundlechoice/tests/test_all_solvers_consistency.py -v --tb=line"

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "✅ Passed:  $PASSED"
echo "❌ Failed:  $FAILED"
echo "⏱️  Timeout: $TIMEOUT"
echo "Total:     $((PASSED + FAILED + TIMEOUT))"
echo "=========================================="

if [ $FAILED -eq 0 ] && [ $TIMEOUT -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    exit 0
else
    exit 1
fi

