# Instructions for Local Agent: MPI Timing Comparison

## Current Situation

We are running a side-by-side comparison of two code versions to verify that our timing fixes work correctly and that unaccounted time is eliminated.

## The Problem We Fixed

Previously, the timing statistics showed significant "unaccounted time" (40-47% of total runtime). This was because:
1. **Feature branch**: `mpi_gather` was measuring total wall-clock time including computation, causing double-counting
2. **Main branch**: Computation time inside `compute_gathered_features()` and `compute_gathered_errors()` was not tracked separately

## What We Fixed

### Both Branches:
- **Broke out computation vs gather time**: Now tracking `compute_features`, `compute_errors`, `gather_features`, `gather_errors` separately
- **Fixed `mpi_gather` calculation**: Now only sums gather operations (`gather_bundles + gather_features + gather_errors`), not total time
- **Added iteration overhead tracking**: Captures logger calls, loop overhead, etc.
- **Added rank verification timing**: Included in `init_time`

### Expected Result:
- **Unaccounted time should be ~0%** (or very small, just Python overhead)
- All operations are now properly tracked

## Current Jobs

Two jobs have been submitted with **identical parameters** (same seed=42, same problem size):

1. **Main branch job**: `4004614`
   - Output: `slurm-mpi-test-main-4004614.out`
   - Error: `slurm-mpi-test-main-4004614.err`
   - Location: `/scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/`

2. **Feature branch job**: `4004615`
   - Output: `slurm-mpi-test-feature-4004615.out`
   - Error: `slurm-mpi-test-feature-4004615.err`
   - Location: `/scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/`

## ⚠️ CRITICAL WARNING: DO NOT MESS UP THE BRANCHES

**DO NOT:**
- Switch branches while jobs are running
- Modify code in either branch
- Commit changes to either branch
- Delete or modify the sbatch scripts

**The two branches must remain separate and unchanged until both jobs complete!**

## How You Can Help

1. **Monitor job status**:
   ```bash
   squeue -j 4004614,4004615
   ```

2. **Check if jobs completed**:
   ```bash
   ls -lh /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-*-400461*.out
   ```

3. **When jobs complete, help analyze**:
   - Read both output files
   - Compare timing statistics
   - Verify unaccounted time is ~0% in both
   - Compare total runtime and per-component times
   - Report any discrepancies

4. **If jobs fail**:
   - Check error files: `slurm-mpi-test-*-400461*.err`
   - Report errors but DO NOT fix code until both jobs complete or are cancelled

## What to Look For in Results

### Good Signs:
- Unaccounted time < 1% in both versions
- All timing components sum to ~100% of total time
- Similar total runtime between branches (they should be comparable)
- `mpi_gather` time is small (< 1% of total) in both

### Red Flags:
- Unaccounted time > 5% in either version
- Large discrepancies in total runtime (> 20% difference)
- Missing timing components in output

## Files to Check After Completion

1. Main branch output: `applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.out`
2. Feature branch output: `applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.out`
3. Look for "Timing Statistics:" section in both files

## Test Parameters (Same for Both)

- Agents: 512
- Items: 200
- Features: 6
- Simulations: 10
- MPI Ranks: 160 (8 nodes × 20 tasks/node)
- Seed: 42 (fixed for reproducibility)
- Time limit: 15 minutes

## Next Steps After Jobs Complete

1. Read both output files
2. Extract timing statistics
3. Compare:
   - Total runtime
   - Unaccounted time (should be ~0%)
   - Per-component breakdown
   - `mpi_gather` time
4. Report findings

