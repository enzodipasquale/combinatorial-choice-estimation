# MPI Performance Test

This folder contains a small-scale test to debug MPI gather performance issues.

## Purpose

Test MPI communication optimizations (buffer-based vs pickle-based) with a smaller problem size to identify bottlenecks.

## Test Configuration

- **Problem**: Quadratic Supermodular (similar structure to QuadKnapsack)
- **Agents**: 256
- **Items**: 100
- **Features**: 6 (2 modular agent + 2 modular item + 2 quadratic item)
- **Simulations**: 10
- **MPI Ranks**: 160 (8 nodes × 20 tasks/node)
- **Time Limit**: 2 minutes

## Files

- `run_estimation.py`: Main estimation script using supermodular factory
- `test_mpi.sbatch`: SLURM batch script
- `README.md`: This file

## Running the Test

```bash
cd experiments/mpi_performance_test
sbatch test_mpi.sbatch
```

## Expected Output

Check the timing statistics in the output file:
- `mpi_gather` time should be significantly reduced compared to pickle-based version
- `mpi_broadcast` time should be near zero
- Total runtime should be faster

## Monitoring

```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with actual job ID)
tail -f slurm-mpi-test-JOBID.out
tail -f slurm-mpi-test-JOBID.err
```

