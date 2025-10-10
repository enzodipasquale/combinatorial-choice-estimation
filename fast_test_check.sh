#!/bin/bash
# Fast test checker - 10s timeout for all tests

TIMEOUT=10
PASSED=0
FAILED=0
SLOW=0

run_test() {
    local name=$1
    local cmd=$2
    
    echo -n "$(date +%H:%M:%S) [$((PASSED + FAILED + SLOW + 1))] $name ... "
    
    if ./timeout_wrapper.sh $TIMEOUT bash -c "$cmd" > /dev/null 2>&1; then
        echo "✅ PASS"
        PASSED=$((PASSED + 1))
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "🐌 SLOW (>${TIMEOUT}s)"
            SLOW=$((SLOW + 1))
        else
            echo "❌ FAIL"
            FAILED=$((FAILED + 1))
        fi
    fi
}

echo "Testing with ${TIMEOUT}s timeout..."
echo ""

# Simple tests
run_test "test_config_loading" "pytest bundlechoice/tests/test_config_loading.py -q"
run_test "test_config_management" "pytest bundlechoice/tests/test_config_management.py -q"
run_test "test_data_manager_simple" "pytest bundlechoice/tests/test_data_manager_simple.py -q"
run_test "test_feature_manager" "pytest bundlechoice/tests/test_feature_manager.py -q"

# MPI tests
run_test "test_data_manager_mpi" "mpirun -n 4 pytest bundlechoice/tests/test_data_manager_mpi.py -q"
run_test "test_edge_cases_validation" "mpirun -n 2 pytest bundlechoice/tests/test_edge_cases_validation.py -q"
run_test "test_feature_basics" "mpirun -n 4 pytest bundlechoice/tests/test_feature_basics.py -q"
run_test "test_greedy_caching_comparison" "mpirun -n 4 pytest bundlechoice/tests/test_greedy_caching_comparison.py -q"

# Subproblem tests
run_test "test_greedy_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_greedy_subproblem.py -q"
run_test "test_linear_knapsack_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_linear_knapsack_subproblem.py -q"
run_test "test_quadratic_supermodular_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_quadratic_supermodular_subproblem.py -q"

# Row generation
run_test "test_row_generation_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_greedy.py -q"
run_test "test_row_generation_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_linear_knapsack.py -q"
run_test "test_row_generation_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_plain_single_item.py -q"
run_test "test_row_generation_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_quadratic_supermodular.py -q"

# Row generation 1slack
run_test "test_row_generation_1slack_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_greedy.py -q"
run_test "test_row_generation_1slack_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_linear_knapsack.py -q"
run_test "test_row_generation_1slack_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_plain_single_item.py -q"
run_test "test_row_generation_1slack_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_quadratic_supermodular.py -q"

# Ellipsoid
run_test "test_ellipsoid_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_greedy.py -q"
run_test "test_ellipsoid_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_linear_knapsack.py -q"
run_test "test_ellipsoid_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_plain_single_item.py -q"
run_test "test_ellipsoid_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_quadratic_supermodular.py -q"

# Integration tests
run_test "test_all_solvers_consistency" "mpirun -n 10 pytest bundlechoice/tests/test_all_solvers_consistency.py -q"

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo "✅ Passed:  $PASSED"
echo "❌ Failed:  $FAILED"
echo "🐌 Slow:    $SLOW (>${TIMEOUT}s)"
echo "Total:     $((PASSED + FAILED + SLOW))"

