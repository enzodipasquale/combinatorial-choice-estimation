# PROMPT FOR HPC AGENT

You are helping with an MPI performance comparison between two code branches. 

## CRITICAL: DO NOT MODIFY EITHER BRANCH
- Two jobs are running: main branch (4004614) and feature branch (4004615)
- DO NOT switch branches, commit changes, or modify code until both jobs complete
- The branches must remain separate and unchanged

## Your Tasks:
1. Monitor job status: `squeue -j 4004614,4004615`
2. When jobs complete, read the output files:
   - Main: applications/combinatorial_auction_v2/slurm-mpi-test-main-4004614.out
   - Feature: applications/combinatorial_auction_v2/slurm-mpi-test-feature-4004615.out
3. Compare timing statistics:
   - Check if "Unaccounted time" is ~0% in both (this is what we fixed)
   - Compare total runtime, mpi_gather time, and per-component breakdown
   - Report any issues or discrepancies

## What We Fixed:
- Broke out computation vs gather time (compute_features, gather_features, etc.)
- Fixed mpi_gather to only sum gather operations, not include computation
- Added iteration overhead tracking
- Expected: unaccounted time should be ~0% (was 40-47% before)

## If Jobs Fail:
- Check error files but DO NOT fix code
- Report errors and wait for instructions

## Full Instructions:
See experiments/mpi_performance_test/LOCAL_AGENT_INSTRUCTIONS.md for complete details.
This file is available on BOTH branches (main and feature/mpi-gather-optimization).

