# TASKS FOR LOCAL AGENT

## Your Role
You run code **locally** (not on SLURM). You can help by:
- Testing code locally with small MPI ranks (2-4 ranks)
- Verifying timing fixes work correctly
- Debugging issues before they hit SLURM
- Running quick validation tests

## Current Status
- Two SLURM jobs are running: main (4004614) and feature (4004615)
- These are large-scale tests (160 ranks, 8 nodes)
- You can run **small-scale local tests** to verify the fixes work

## Your Tasks

### 1. Test Timing Fixes Locally (Both Branches)

#### Test Main Branch:
```bash
cd /scratch/ed2189/combinatorial-choice-estimation
git checkout main

# Run small local test (2-4 ranks)
mpirun -n 4 python experiments/mpi_performance_test/run_estimation.py
```

**What to check:**
- Does it run without errors?
- In the "Timing Statistics:" section, is "Unaccounted time" ~0%?
- Are all timing components present (gather_bundles, gather_features, compute_features, etc.)?

#### Test Feature Branch:
```bash
cd /scratch/ed2189/combinatorial-choice-estimation
git checkout feature/mpi-gather-optimization

# Run small local test (2-4 ranks)
mpirun -n 4 python experiments/mpi_performance_test/run_estimation.py
```

**What to check:**
- Same as above
- Compare unaccounted time between branches (should both be ~0%)

### 2. Verify Timing Components Are Tracked

For both branches, check that the output includes:
- `gather_bundles`
- `gather_features`
- `gather_errors`
- `compute_features`
- `compute_errors`
- `mpi_gather` (should be sum of gather operations)
- `iteration_overhead`
- `Unaccounted time` (should be ~0% or very small)

### 3. Quick Validation Test

Create a simple test script to verify timing adds up:
```python
# After running, check that:
# total_time ≈ init_time + sum(all_component_times)
# unaccounted_time = total_time - sum(all_tracked_times) ≈ 0
```

### 4. Report Findings

Report:
- ✅ Both branches run successfully locally
- ✅ Unaccounted time is ~0% in both
- ✅ All timing components are present
- ⚠️ Any errors or issues found
- ⚠️ Any discrepancies between branches

## ⚠️ CRITICAL REMINDER
**DO NOT modify code in either branch!**
- Test both branches as-is
- Report issues but don't fix them
- Wait for instructions before making changes

## If You Find Issues

1. **Syntax errors**: Report immediately
2. **Timing issues**: Note which branch and what's wrong
3. **Missing components**: List what's missing
4. **Unaccounted time > 1%**: Report the value

## Expected Local Test Results

With 2-4 ranks, the test should:
- Complete in < 1 minute
- Show unaccounted time < 1%
- Show all timing components
- Run without errors

## Files to Use
- Test script: `experiments/mpi_performance_test/run_estimation.py`
- This works on both branches (same script, different code versions)
