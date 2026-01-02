# Coordination Between HPC Agent and Local Agent

## How This Works

- **HPC Agent**: On HPC cluster, can ONLY use SLURM jobs (cannot run scripts on login node)
- **Local Agent**: On user's laptop, CAN run scripts locally
- **User**: Copies messages between agents via code boxes

## Current Status

- **Feature branch**: Combined gather optimization implemented (1 gather instead of 3)
- **Goal**: Verify speedup, especially at large sizes (XL: 512×200)
- **Previous finding**: Feature branch was 1.7% slower at XL size - need to verify if combined gather fixes this

## Latest Change (Feature Branch)

- **Combined gather optimization**: Gather bundles once, compute features/errors on root
- **Reduces MPI operations**: 3 gathers → 1 gather
- **Expected benefit**: Should help at large scales where communication dominates
- **Commit**: `d6e580d` - "opt: implement combined gather optimization"

## Messages

Agents communicate via code box messages that user copy-pastes.
See code boxes in conversation for current messages.

