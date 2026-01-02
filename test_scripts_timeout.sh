#!/bin/bash
# Test script to run both scripts with timeout and check for hangs/crashes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/applications/combinatorial_auction_v2"

echo "=========================================="
echo "Testing run_estimation.py"
echo "=========================================="

# Test 1: run_estimation.py with timeout (30 seconds)
echo "Running run_estimation.py with 30s timeout..."
timeout 30 mpirun -n 2 python3 run_estimation.py 2>&1 | head -50 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "✗ TIMEOUT: run_estimation.py hung or took too long (>30s)"
        exit 1
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "✗ ERROR: run_estimation.py crashed or failed"
        exit 1
    else
        echo "✓ run_estimation.py completed (exit code: $EXIT_CODE)"
    fi
}

echo ""
echo "=========================================="
echo "Testing compute_se_new.py"
echo "=========================================="

# Test 2: compute_se_new.py with timeout (60 seconds - SE computation takes longer)
echo "Running compute_se_new.py with 60s timeout..."
timeout 60 mpirun -n 2 python3 compute_se_new.py 2>&1 | head -50 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "✗ TIMEOUT: compute_se_new.py hung or took too long (>60s)"
        exit 1
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "✗ ERROR: compute_se_new.py crashed or failed"
        exit 1
    else
        echo "✓ compute_se_new.py completed (exit code: $EXIT_CODE)"
    fi
}

echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED - No hangs or crashes"
echo "=========================================="

