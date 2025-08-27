#!/bin/bash

# Script to run benchmarks without allowing MacBook to sleep
# Usage: ./run_benchmark_no_sleep.sh [greedy|supermod|knapsack|plain]

if [ $# -eq 0 ]; then
    echo "Usage: $0 [greedy|supermod|knapsack|plain]"
    echo "Example: $0 greedy"
    exit 1
fi

BENCHMARK_TYPE=$1

echo "🚀 Starting $BENCHMARK_TYPE benchmark with comprehensive sleep prevention..."
echo "💤 Your MacBook will NOT sleep for any reason during this process"
echo "🛡️  Using: caffeinate -i -s -d (prevents idle, system, and display sleep)"
echo "⏰ Started at: $(date)"
echo ""

# Run the benchmark with comprehensive caffeinate to prevent all types of sleep
case $BENCHMARK_TYPE in
    "greedy")
        caffeinate -i -s -d make greedy_benchmark
        ;;
    "supermod")
        caffeinate -i -s -d make supermod_benchmark
        ;;
    "knapsack")
        caffeinate -i -s -d make knapsack_benchmark
        ;;
    "plain")
        caffeinate -i -s -d make plain_benchmark
        ;;
    *)
        echo "❌ Unknown benchmark type: $BENCHMARK_TYPE"
        echo "Valid options: greedy, supermod, knapsack, plain"
        exit 1
        ;;
esac

echo ""
echo "✅ Benchmark completed at: $(date)"
echo "💤 Sleep prevention disabled - your MacBook can now sleep normally"
