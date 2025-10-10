#!/bin/bash
# Verify all remaining tests pass under 10 seconds

TIMEOUT=10
PASSED=0
FAILED=0
SLOW=0

run_test() {
    local name=$1
    local cmd=$2
    
    echo -n "[$((PASSED + FAILED + SLOW + 1))] $name ... "
    start=$(date +%s)
    
    if ./timeout_wrapper.sh $TIMEOUT bash -c "$cmd" > /dev/null 2>&1; then
        end=$(date +%s)
        duration=$((end - start))
        echo "✅ PASS (${duration}s)"
        PASSED=$((PASSED + 1))
    else
        exit_code=$?
        end=$(date +%s)
        duration=$((end - start))
        if [ $exit_code -eq 124 ]; then
            echo "🐌 SLOW (>10s)"
            SLOW=$((SLOW + 1))
        else
            echo "❌ FAIL (${duration}s)"
            FAILED=$((FAILED + 1))
        fi
    fi
}

echo "Verifying all tests with 10s timeout..."
echo ""

run_test "test_config_loading" "pytest bundlechoice/tests/test_config_loading.py -q"
run_test "test_config_management" "pytest bundlechoice/tests/test_config_management.py -q"
run_test "test_data_manager_simple" "pytest bundlechoice/tests/test_data_manager_simple.py -q"
run_test "test_feature_manager" "pytest bundlechoice/tests/test_feature_manager.py -q"
run_test "test_data_manager_mpi" "mpirun -n 4 pytest bundlechoice/tests/test_data_manager_mpi.py -q"
run_test "test_greedy_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_greedy_subproblem.py -q"
run_test "test_linear_knapsack_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_linear_knapsack_subproblem.py -q"
run_test "test_quadratic_supermodular_subproblem" "mpirun -n 10 pytest bundlechoice/tests/test_quadratic_supermodular_subproblem.py -q"
run_test "test_row_generation_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_greedy.py -q"
run_test "test_row_generation_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_linear_knapsack.py -q"
run_test "test_row_generation_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_plain_single_item.py -q"
run_test "test_row_generation_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_quadratic_supermodular.py -q"
run_test "test_row_generation_1slack_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_greedy.py -q"
run_test "test_row_generation_1slack_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_linear_knapsack.py -q"
run_test "test_row_generation_1slack_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_plain_single_item.py -q"
run_test "test_row_generation_1slack_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_row_generation_1slack_quadratic_supermodular.py -q"
run_test "test_ellipsoid_greedy" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_greedy.py -q"
run_test "test_ellipsoid_linear_knapsack" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_linear_knapsack.py -q"
run_test "test_ellipsoid_plain_single_item" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_plain_single_item.py -q"
run_test "test_ellipsoid_quadratic_supermodular" "mpirun -n 10 pytest bundlechoice/tests/test_estimation_ellipsoid_quadratic_supermodular.py -q"
run_test "test_all_solvers_consistency" "mpirun -n 10 pytest bundlechoice/tests/test_all_solvers_consistency.py -q"

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo "✅ Passed:  $PASSED / $((PASSED + FAILED + SLOW))"
echo "❌ Failed:  $FAILED"
echo "🐌 Slow:    $SLOW (>10s)"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $SLOW -eq 0 ]; then
        echo "🎉 ALL TESTS PASS UNDER 10 SECONDS!"
    else
        echo "⚠️  All tests pass, but $SLOW test(s) exceed 10s"
    fi
    exit 0
else
    echo "❌ $FAILED test(s) failed"
    exit 1
fi

