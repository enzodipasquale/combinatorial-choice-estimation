# Coordination Between HPC Agent and Local Agent

## How This Works

- **HPC Agent**: On HPC cluster, can ONLY use SLURM jobs (cannot run scripts on login node)
- **Local Agent**: On user's laptop, CAN run scripts locally
- **User**: Copies messages between agents via code boxes

## Current Status

- Two SLURM jobs running: main (4004614) and feature (4004615)
- Both testing same problem (seed=42, 512 agents, 200 items, 6 features, 10 simuls)
- Goal: Verify unaccounted time is ~0% in both (was 40-47% before fixes)

## What We Fixed

- Broke out computation vs gather time separately
- Fixed `mpi_gather` to only sum gather operations (not computation)
- Added iteration overhead tracking
- Expected: unaccounted time should be ~0%

## Messages

Agents communicate via code box messages that user copy-pastes.
See code boxes in conversation for current messages.

