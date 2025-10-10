#!/bin/bash
# Run all tests with appropriate timeouts to ensure nothing hangs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT_WRAPPER="$SCRIPT_DIR/timeout_wrapper.sh"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

total_tests=0
passed_tests=0
failed_tests=0
timeout_tests=0

run_test() {
    local test_file=$1
    local timeout=$2
    local mpi_ranks=${3:-1}
    local test_name=$(basename "$test_file" .py)
    
    total_tests=$((total_tests + 1))
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[$total_tests] Running: $test_name${NC}"
    echo -e "${BLUE}    Timeout: ${timeout}s | MPI ranks: $mpi_ranks${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $mpi_ranks -eq 1 ]; then
        cmd="pytest $test_file -v --tb=short"
    else
        cmd="mpirun -n $mpi_ranks pytest $test_file -v --tb=short"
    fi
    
    # Use timeout wrapper or gtimeout if available, otherwise use Perl-based timeout
    if command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout $timeout"
    elif [ -x "$TIMEOUT_WRAPPER" ]; then
        timeout_cmd="$TIMEOUT_WRAPPER $timeout"
    else
        # Fallback: use Perl one-liner for timeout (works on macOS)
        timeout_cmd="perl -e 'alarm shift; exec @ARGV' $timeout"
    fi
    
    if $timeout_cmd bash -c "$cmd"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        passed_tests=$((passed_tests + 1))
        return 0
    else
        exit_code=$?
        if [ $exit_code -eq 124 ] || [ $exit_code -eq 142 ]; then
            echo -e "${RED}⏱️  TIMEOUT: $test_name (exceeded ${timeout}s)${NC}"
            timeout_tests=$((timeout_tests + 1))
        else
            echo -e "${RED}❌ FAILED: $test_name${NC}"
            failed_tests=$((failed_tests + 1))
        fi
        return 1
    fi
}

echo -e "${YELLOW}Starting test suite with timeout protection...${NC}\n"

# ============================================================================
# Category 1: Simple tests (no MPI, 30s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 1: Simple Tests (30s timeout)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_config_loading.py" 30 1
run_test "bundlechoice/tests/test_config_management.py" 30 1
run_test "bundlechoice/tests/test_data_manager_simple.py" 30 1
run_test "bundlechoice/tests/test_feature_manager.py" 30 1

# ============================================================================
# Category 2: MPI tests (2-4 ranks, 60s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 2: MPI Tests (60s timeout)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_data_manager_mpi.py" 60 4
run_test "bundlechoice/tests/test_edge_cases_validation.py" 60 2
run_test "bundlechoice/tests/test_feature_basics.py" 60 4
run_test "bundlechoice/tests/test_greedy_caching_comparison.py" 60 4

# ============================================================================
# Category 3: Subproblem tests (10 MPI ranks, 90s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 3: Subproblem Tests (90s timeout, 10 MPI ranks)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_greedy_subproblem.py" 90 10
run_test "bundlechoice/tests/test_linear_knapsack_subproblem.py" 90 10
run_test "bundlechoice/tests/test_quadratic_supermodular_subproblem.py" 90 10

# ============================================================================
# Category 4: Row Generation tests (10 MPI ranks, 120s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 4: Row Generation Tests (120s timeout, 10 MPI ranks)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_estimation_row_generation_greedy.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_linear_knapsack.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_plain_single_item.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_quadratic_supermodular.py" 120 10

# ============================================================================
# Category 5: Row Generation 1-slack tests (10 MPI ranks, 120s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 5: Row Generation 1-Slack Tests (120s timeout, 10 MPI ranks)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_estimation_row_generation_1slack_greedy.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_1slack_linear_knapsack.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_1slack_plain_single_item.py" 120 10
run_test "bundlechoice/tests/test_estimation_row_generation_1slack_quadratic_supermodular.py" 120 10

# ============================================================================
# Category 6: Ellipsoid tests (10 MPI ranks, 90s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 6: Ellipsoid Tests (90s timeout, 10 MPI ranks)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_estimation_ellipsoid_greedy.py" 90 10
run_test "bundlechoice/tests/test_estimation_ellipsoid_linear_knapsack.py" 90 10
run_test "bundlechoice/tests/test_estimation_ellipsoid_plain_single_item.py" 90 10
run_test "bundlechoice/tests/test_estimation_ellipsoid_quadratic_supermodular.py" 90 10

# ============================================================================
# Category 7: Integration & Property tests (slower, 120s timeout)
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CATEGORY 7: Integration & Property Tests (120s timeout, 4 MPI ranks)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"

run_test "bundlechoice/tests/test_property_based.py" 120 4
run_test "bundlechoice/tests/test_integration_comprehensive.py" 120 4
run_test "bundlechoice/tests/test_all_solvers_consistency.py" 120 10

# ============================================================================
# Summary
# ============================================================================
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}TEST SUMMARY${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "Total tests:   $total_tests"
echo -e "${GREEN}Passed:        $passed_tests${NC}"
echo -e "${RED}Failed:        $failed_tests${NC}"
echo -e "${RED}Timeouts:      $timeout_tests${NC}"

if [ $failed_tests -eq 0 ] && [ $timeout_tests -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ SOME TESTS FAILED OR TIMED OUT${NC}"
    exit 1
fi

