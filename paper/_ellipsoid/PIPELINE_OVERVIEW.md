# Experiments Pipeline Overview

## Standardized Structure

All 4 experiment types (greedy, supermod, knapsack, supermodknapsack) now follow a consistent structure:

### Configuration Files
- **config.yaml**: Experiment parameters (dimensions, replications, solver settings)
- **sizes.yaml**: Problem size grid definitions
- **run.py**: Main experiment script (standardized across all experiments)

### Features

1. **Environment Variable Override**: All scripts support `NUM_AGENTS` and `NUM_ITEMS` environment variables
2. **Timing Breakdown**: All scripts capture detailed timing statistics (compute, solve, comm) with percentages
3. **Consistent CSV Output**: All experiments produce the same CSV column structure:
   - Metadata: replication, seed, method, time_s, obj_value, dimensions, sigma, subproblem
   - Timing: timing_compute, timing_solve, timing_comm (and percentages)
   - Parameters: theta_true_0..K, theta_0..K

4. **Error Handling**: All scripts handle errors gracefully and log ERROR rows

### Running Experiments

**Single experiment:**
```bash
sbatch run_greedy.sbatch
sbatch run_supermod.sbatch
sbatch run_knapsack.sbatch
sbatch run_supermodknapsack.sbatch
```

**All experiments:**
```bash
./run_all_experiments.sh
```

**With custom timeout:**
The pipeline supports `--timeout` parameter (in seconds) for each size:
```bash
python run_experiment.py greedy --mpi 10 --timeout 240
```

### Configuration Options

**config.yaml** supports:
- `dimensions`: num_agents, num_items, num_features, num_simuls
- `num_replications`: Number of replications (default: 10)
- `base_seed`: Starting seed for random number generation
- `sigma`: Error standard deviation
- `row_generation`: Solver settings (max_iters, tolerance_optimality, etc.)
- `ellipsoid`: Ellipsoid solver settings

**sizes.yaml** supports two formats:
- **Named sizes**: `sizes: {small: {...}, medium: {...}}`
- **Grid**: `sizes_grid: {agents: [5,10,20], items: [10,20]}`

### Output Structure

Each experiment creates timestamped output directories:
```
outputs/experiment_YYYYMMDD_HHMMSS/
  - results_raw.csv       # Raw results from all sizes/replications
  - results_summary.csv   # Aggregated statistics
  - tables.tex           # LaTeX tables for paper
  - config.yaml          # Configuration used
  - sizes.yaml           # Size definitions used
```

### Flexibility

- **Size override**: Set `NUM_AGENTS` and `NUM_ITEMS` environment variables
- **Replications**: Configure in `config.yaml`
- **Timeout**: Set per-size timeout via `--timeout` flag
- **MPI processes**: Configure via `--mpi` flag (default: 10)

