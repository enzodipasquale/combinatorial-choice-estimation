# Numerical Experiments for Paper

This directory contains scripts for running numerical experiments comparing different estimation methods (Row Generation, Row Generation 1-Slack, and Ellipsoid).

## Structure

Each experiment type has its own directory:
- `greedy/` - Greedy subproblem
- `supermod/` - Quadratic Supermodular Network subproblem
- `knapsack/` - Linear Knapsack subproblem
- `supermodknapsack/` - Quadratic Knapsack subproblem

Each directory contains:
- `config.yaml` - Experiment configuration (number of agents, items, features, etc.)
- `sizes.yaml` - Problem size configurations (small, medium, large)
- `run.py` - Main experiment script
- `results.csv` - Results from running experiments (appended on each replication)

## Running Experiments

### Quick Start: Complete Pipeline

Use the main pipeline script to run all sizes and generate outputs:

```bash
python experiments_paper/run_experiment.py greedy --mpi 10
```

This will:
1. Run experiments for all configured sizes (from `sizes.yaml`)
2. Copy raw results to a timestamped output directory
3. Generate summary statistics
4. Generate LaTeX tables
5. Save all outputs in `experiments_paper/outputs/greedy_YYYYMMDD_HHMMSS/`

### Manual Single Run

To run a single experiment for one size:

1. Configure parameters in `config.yaml`:
   - `num_agents`: Number of agents
   - `num_items`: Number of items
   - `num_features`: Number of features
   - `num_replications`: Number of replications to run
   - `base_seed`: Starting seed for random number generation
   - `solver_precision`: Precision for ellipsoid solver (automatically computes iterations)

2. Run the experiment:
```bash
mpirun -n 10 python experiments_paper/<experiment_type>/run.py
```

### Running Multiple Sizes

To run experiments across multiple sizes:

1. Configure sizes in `sizes.yaml`:
```yaml
sizes:
  small:
    num_agents: 5
    num_items: 5
  medium:
    num_agents: 10
    num_items: 10
  large:
    num_agents: 20
    num_items: 20
```

2. Run all sizes:
```bash
python experiments_paper/run_all_sizes.py greedy --mpi 10
```

Or override sizes via environment variables:
```bash
NUM_AGENTS=20 NUM_ITEMS=20 mpirun -n 10 python experiments_paper/greedy/run.py
```

### Generating Tables Only

To generate LaTeX tables from existing results:

```bash
python experiments_paper/run_experiment.py greedy --skip-run
```

## Output Format

### Results CSV (`results.csv`)

Each row contains:
- **Metadata**: `replication`, `seed`, `method`, `time_s`, `obj_value`, `num_agents`, `num_items`, `num_features`, `num_simuls`, `sigma`, `subproblem`
- **Timing breakdown**: `timing_compute`, `timing_solve`, `timing_comm`, and their percentages
  - For row generation: `timing_compute` = pricing time, `timing_solve` = master problem time
  - For ellipsoid: `timing_compute` = gradient computation time, `timing_solve` = ellipsoid update time
- **True parameters**: `theta_true_0`, `theta_true_1`, ..., `theta_true_{K-1}`
- **Estimated parameters**: `theta_0`, `theta_1`, ..., `theta_{K-1}`

### Summary CSV (`results_summary.csv`)

Contains aggregated statistics per method:
- `runtime_mean`, `runtime_std`: Average runtime over replications
- `rmse_0`, `rmse_1`, ..., `rmse_{K-1}`: Root Mean Squared Error per parameter
- `bias_0`, `bias_1`, ..., `bias_{K-1}`: Bias (mean error) per parameter

### LaTeX Tables (`tables.tex`)

Generated automatically with:
- Slide 1: Experiment description with utility function, features, and error model
- Slide 2: Comparison table showing runtime, RMSE, and Bias for all methods and sizes
- Professional formatting suitable for top-tier venues

To generate tables:
```bash
python experiments_paper/generate_latex_tables.py experiments_paper --experiment Greedy --output tables.tex
```

### Timestamped Outputs

The `run_experiment.py` script creates organized outputs in `experiments_paper/outputs/experiment_YYYYMMDD_HHMMSS/`:
- `results_raw.csv`: Original results
- `results_summary.csv`: Aggregated statistics
- `tables.tex`: LaTeX tables
- `config.yaml`: Configuration used
- `sizes.yaml`: Size definitions
- `README.md`: Metadata and information

## Notes

- All three methods (row_generation, row_generation_1slack, ellipsoid) are run on the same problem instance for fair comparison
- Theta values are compared to ensure consistency (printed during run)
- Ellipsoid solver uses warm start from row generation solution for better convergence
- Solver precision is used to automatically compute iteration counts (no need to specify num_iters)

