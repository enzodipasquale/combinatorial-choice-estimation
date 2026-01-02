# PROMPT FOR HPC AGENT

You are helping with an MPI performance comparison between two code branches.

## The Setup

**You (HPC Agent)**:
- On HPC cluster login node
- **CANNOT run scripts directly** (login node restrictions)
- **CAN ONLY submit SLURM jobs** (already done)
- Can read SLURM output files when jobs complete

**Local Agent**:
- On user's laptop
- **CAN run scripts locally** (testing both branches now)
- Will report local test findings

## CRITICAL: DO NOT MODIFY EITHER BRANCH
- Two SLURM jobs are running: main (4004614) and feature (4004615)
- DO NOT switch branches, commit changes, or modify code until both jobs complete
- The branches must remain separate and unchanged

## Your Tasks:

### 1. Monitor Job Status
```bash
squeue -j 4004614,4004615
```

### 2. When Jobs Complete
Read the output files:
- Main: `applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.out`
- Feature: `applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.out`

### 3. Extract Timing Statistics
Look for "Timing Statistics:" section and check:
- ✅ **Unaccounted time is ~0%** (this is what we fixed!)
- ✅ All components present (gather_bundles, gather_features, compute_features, etc.)
- ✅ `mpi_gather` is small (< 1% of total)

### 4. Compare Results
- Total runtime comparison
- Unaccounted time (should be ~0% in both)
- Per-component breakdown
- `mpi_gather` time
- Report any discrepancies

### 5. If Jobs Fail
- Check error files: `slurm-mpi-test-*-400461*.err`
- Report errors but **DO NOT fix code**
- Wait for coordination

## What We Fixed

- Broke out computation vs gather time separately
- Fixed `mpi_gather` to only sum gather operations (not computation)
- Added iteration overhead tracking
- Expected: unaccounted time should be ~0% (was 40-47% before)

## Current Jobs

- **Main branch**: Job 4004614 (160 ranks, 8 nodes)
- **Feature branch**: Job 4004615 (160 ranks, 8 nodes)
- Both use same parameters (seed=42, same problem size)

## Coordination

- Local agent is testing both branches locally (you can't do this)
- You monitor SLURM jobs and analyze results when they complete
- Report findings and coordinate with local agent's results
