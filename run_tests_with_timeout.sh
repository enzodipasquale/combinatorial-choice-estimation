#!/bin/bash
# Test runner script with timeout wrappers
# Usage: ./run_tests_with_timeout.sh [test_pattern]
set -e

# Default timeout values (in seconds)
FAST_TEST_TIMEOUT=10
MEDIUM_TEST_TIMEOUT=60
SLOW_TEST_TIMEOUT=120
MPI_TEST_TIMEOUT=180

# Function to get timeout for a test
get_timeout() {
    local test_name=$1
    
    # Fast tests (simple unit tests)
    case $test_name in
        test_config_loading|test_config_management|test_data_manager_simple|test_feature_manager)
            echo $FAST_TEST_TIMEOUT
            ;;
        # Medium tests (estimation tests)
        test_estimation_ellipsoid_greedy|test_estimation_ellipsoid_linear_knapsack|test_estimation_ellipsoid_plain_single_item|test_estimation_row_generation_greedy|test_estimation_row_generation_linear_knapsack|test_estimation_row_generation_plain_single_item)
            echo $MEDIUM_TEST_TIMEOUT
            ;;
        # Slow tests (complex estimation and subproblem tests)
        test_estimation_ellipsoid_quadratic_supermodular|test_estimation_row_generation_quadratic_supermodular|test_greedy_subproblem|test_linear_knapsack_subproblem|test_quadratic_supermodular_subproblem)
            echo $SLOW_TEST_TIMEOUT
            ;;
        # MPI tests (need more time for parallel execution)
        test_data_manager_mpi)
            echo $MPI_TEST_TIMEOUT
            ;;
        # Default
        *)
            echo $MEDIUM_TEST_TIMEOUT
            ;;
    esac
}

# Function to check if test needs MPI
needs_mpi() {
    local test_name=$1
    
    case $test_name in
        test_estimation_ellipsoid_greedy|test_estimation_ellipsoid_linear_knapsack|test_estimation_ellipsoid_plain_single_item|test_estimation_ellipsoid_quadratic_supermodular|test_estimation_row_generation_greedy|test_estimation_row_generation_linear_knapsack|test_estimation_row_generation_plain_single_item|test_estimation_row_generation_quadratic_supermodular|test_data_manager_mpi)
            return 0  # true
            ;;
        *)
            return 1  # false
            ;;
    esac
}

run_test_with_timeout() {
    local test_file=$1
    local test_name=$(basename "$test_file" .py)
    local timeout=$(get_timeout "$test_name")
    
    echo "Running $test_name with ${timeout}s timeout..."
    
    if needs_mpi "$test_name"; then
        # MPI tests need special handling with 10 processes
        ./timeout_wrapper.sh $timeout mpirun -np 10 python3 -m pytest "$test_file" -v
    else
        # Regular tests
        ./timeout_wrapper.sh $timeout python3 -m pytest "$test_file" -v
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ $test_name passed"
    else
        echo "❌ $test_name failed or timed out"
        return 1
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # Run all tests
    echo "Running all tests with timeout wrappers..."
    for test_file in bundlechoice/tests/test_*.py; do
        run_test_with_timeout "$test_file"
    done
else
    # Run specific test pattern
    pattern=$1
    echo "Running tests matching pattern: $pattern"
    for test_file in bundlechoice/tests/test_*${pattern}*.py; do
        if [ -f "$test_file" ]; then
            run_test_with_timeout "$test_file"
        fi
    done
fi

echo "All tests completed!" 