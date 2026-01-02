# TASKS FOR LOCAL AGENT

## Current Status
- Two jobs submitted: main (4004614) and feature (4004615)
- Both jobs are currently PENDING in the queue
- Jobs will run with identical parameters (seed=42, same problem size)

## Your Tasks While Waiting

### 1. Monitor Job Status
```bash
# Check job status periodically
squeue -j 4004614,4004615

# Check if output files appear
ls -lh /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-*-400461*.out
```

### 2. When Jobs Start Running
- Monitor progress by tailing output files:
```bash
# Main branch
tail -f /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.out

# Feature branch  
tail -f /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.out
```

### 3. When Jobs Complete - Analysis Tasks

#### A. Check for Errors
```bash
# Check error files
cat /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.err
cat /scratch/ed2189/combinatorial-choice-estimation/applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.err
```

#### B. Extract Timing Statistics
For each output file, find the "Timing Statistics:" section and extract:
- Total time
- Unaccounted time (should be ~0%)
- Component breakdown:
  - pricing
  - mpi_gather
  - gather_bundles, gather_features, gather_errors
  - compute_features, compute_errors
  - master_prep, master_update, master_optimize
  - mpi_broadcast
  - iteration_overhead

#### C. Create Comparison Report
Compare the two versions:
1. **Unaccounted time**: Should be ~0% in both (this is what we fixed)
2. **Total runtime**: Should be similar (within 10-20%)
3. **mpi_gather time**: Should be small (< 1% of total) in both
4. **Per-component times**: Compare each component between versions
5. **Any discrepancies**: Note any significant differences

#### D. Report Format
Create a summary with:
```
=== TIMING COMPARISON REPORT ===

Main Branch (4004614):
- Total time: X.XXs
- Unaccounted time: X.XXs (X.X%)
- mpi_gather: X.XXs (X.X%)
- [other components...]

Feature Branch (4004615):
- Total time: X.XXs
- Unaccounted time: X.XXs (X.X%)
- mpi_gather: X.XXs (X.X%)
- [other components...]

Comparison:
- Unaccounted time fixed: [YES/NO - should be ~0% in both]
- Runtime difference: X.X% [main vs feature]
- Key findings: [list any issues or discrepancies]
```

## ⚠️ CRITICAL REMINDER
**DO NOT modify either branch until both jobs complete and analysis is done!**

## Files to Check
- Main output: `applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.out`
- Feature output: `applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.out`
- Main error: `applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.err`
- Feature error: `applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.err`

