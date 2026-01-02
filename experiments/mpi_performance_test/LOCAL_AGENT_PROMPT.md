# PROMPT FOR LOCAL AGENT

You are helping with an MPI performance comparison between two code branches.

## The Setup

**HPC Agent**: 
- On HPC cluster login node
- **CANNOT run scripts directly** (login node restrictions)
- **CAN ONLY submit SLURM jobs** (already done: jobs 4004614 and 4004615)
- Will analyze SLURM results when jobs complete

**You (Local Agent)**:
- On user's laptop
- **CAN run scripts directly** (no SLURM needed)
- **Your job: Test both branches locally** to verify timing fixes work

## CRITICAL: DO NOT MODIFY EITHER BRANCH
- Test both branches as-is
- Report issues but don't fix them
- Wait for coordination before making changes

## Your Tasks - START NOW:

### 1. Test Main Branch Locally
```bash
cd /path/to/combinatorial-choice-estimation
git checkout main
mpirun -n 4 python experiments/mpi_performance_test/run_estimation.py
```

**Check:**
- ✅ Runs without errors?
- ✅ "Unaccounted time" is ~0%? (this is what we fixed!)
- ✅ All timing components present? (gather_bundles, gather_features, compute_features, etc.)

### 2. Test Feature Branch Locally
```bash
cd /path/to/combinatorial-choice-estimation
git checkout feature/mpi-gather-optimization
mpirun -n 4 python experiments/mpi_performance_test/run_estimation.py
```

**Check:**
- ✅ Same as above
- ✅ Compare unaccounted time (should both be ~0%)

### 3. Verify Timing Components
Both branches should show in "Timing Statistics:":
- `gather_bundles`, `gather_features`, `gather_errors`
- `compute_features`, `compute_errors`
- `mpi_gather` (should be sum of gather operations only)
- `iteration_overhead`
- `Unaccounted time` (should be ~0% or very small)

### 4. Report Findings
Report:
- ✅ Both branches run successfully locally
- ✅ Unaccounted time is ~0% in both
- ✅ All timing components are present
- ⚠️ Any errors or issues found
- ⚠️ Any discrepancies between branches

## What We Fixed

- Broke out computation vs gather time separately
- Fixed `mpi_gather` to only sum gather operations (not computation)
- Added iteration overhead tracking
- Expected: unaccounted time should be ~0% (was 40-47% before)

## Expected Results

With 2-4 ranks locally:
- Should complete in < 1 minute
- Unaccounted time < 1%
- All timing components present
- No errors

## Files to Use
- Test script: `experiments/mpi_performance_test/run_estimation.py`
- Same script works on both branches (different code versions)

**START TESTING NOW!**

