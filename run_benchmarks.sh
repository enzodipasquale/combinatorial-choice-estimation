#!/bin/bash

# Script to run benchmarks without allowing MacBook to sleep
# Usage: 
#   ./run_benchmarks.sh [greedy|supermod|knapsack|plain]  # Run specific benchmark
#   ./run_benchmarks.sh --all                              # Run all benchmarks

show_usage() {
    echo "Usage: $0 [greedy|supermod|knapsack|plain|--all]"
    echo ""
    echo "Examples:"
    echo "  $0 greedy       # Run greedy benchmark only"
    echo "  $0 --all        # Run all benchmarks"
    exit 1
}

if [ $# -eq 0 ]; then
    show_usage
fi

BENCHMARK_TYPE=$1

echo "🚀 Starting benchmark(s) with comprehensive sleep prevention..."
echo "💤 Your MacBook will NOT sleep for any reason during this process"
echo "🛡️  Using: caffeinate -i -s -d (prevents idle, system, and display sleep)"
echo "⏰ Started at: $(date)"
echo ""

# Run the benchmark with comprehensive caffeinate to prevent all types of sleep
case $BENCHMARK_TYPE in
    "--all")
        caffeinate -i -s -d bash -c "
        make greedy_benchmark
        make supermod_benchmark  
        make knapsack_benchmark
        "
        ;;
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
        show_usage
        ;;
esac

echo ""
echo "✅ Benchmark completed at: $(date)"
echo "💤 Sleep prevention disabled - your MacBook can now sleep normally"
