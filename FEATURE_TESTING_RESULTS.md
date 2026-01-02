# Feature Testing Results - Systematic Analysis

## Goal
Identify which features from experimental branch are worth keeping vs main branch at 12+ ranks.

## Baseline
- **Main branch (12 ranks, XL)**: 32.19s
- **Feature branch baseline (match main)**: ~32-43s (inconsistent, needs more runs)

## Features to Test

### 1. Hash-based Constraint Naming
- **Current**: Binary string `f"rowgen_{idx}_bundle_{bundle_binary}"`
- **Feature**: Hash-based `f"rowgen_{idx}_b{abs(bundle_hash) % 1000000000}"`
- **Benefit**: Avoids Gurobi 255 char limit
- **Cost**: Hash computation overhead
- **Status**: Testing...

### 2. broadcast_array_with_flag
- **Current**: `broadcast_from_root()` (pickle-based)
- **Feature**: `broadcast_array_with_flag()` (buffer-based)
- **Benefit**: Faster than pickle, combines theta + stop flag
- **Cost**: Pre-allocation requirement
- **Status**: Testing...

### 3. Detailed Timing Breakdown
- **Current**: Simple timing (pricing, mpi_gather, master_prep, etc.)
- **Feature**: Detailed (gather_bundles, gather_features, compute_features separately)
- **Benefit**: Better diagnostics
- **Cost**: More timing_dict operations
- **Status**: Known expensive (_log_timing_summary disabled)

## Findings So Far

1. **Timing Summary is EXPENSIVE**: Disabling `_log_timing_summary` improved from 40s+ to ~32s
2. **Rank Verification adds overhead**: Removing it helps
3. **Hash computation**: Need to test if it's faster than binary string concatenation

## Next Steps

1. Run baseline 3-5 times to get consistent average
2. Test each feature individually
3. Keep only features that don't slow down at 12 ranks
4. If no features help, recommend keeping main branch

