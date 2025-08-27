#!/bin/bash

# Run all benchmarks with comprehensive sleep prevention
caffeinate -i -s -d bash -c "
make greedy_benchmark
make supermod_benchmark  
make knapsack_benchmark
"
