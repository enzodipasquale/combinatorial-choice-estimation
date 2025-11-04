# Inversion Experiments for Paper

This directory contains scripts for running numerical experiments comparing naive (biased) vs IV (unbiased) estimation methods with endogeneity.

## Structure

Each experiment type has its own directory:
- `greedy_naive/` - Greedy subproblem with naive estimation (ignores endogeneity)
- `greedy_iv/` - Greedy subproblem with IV estimation (modular inversion + IV regression)
- `supermod_naive/` - Quadratic Supermodular Network with naive estimation
- `supermod_iv/` - Quadratic Supermodular Network with IV estimation
- `knapsack_naive/` - Linear Knapsack with naive estimation
- `knapsack_iv/` - Linear Knapsack with IV estimation
- `quadknapsack_naive/` - Quadratic Knapsack with naive estimation
- `quadknapsack_iv/` - Quadratic Knapsack with IV estimation

Each directory contains:
- `config.yaml` - Experiment configuration
- `sizes.yaml` - Problem size configurations
- `run.py` - Main experiment script
- `results.csv` - Results from running experiments (appended on each replication)

## Running Experiments

### Quick Start: Complete Pipeline

Use the main pipeline script to run all sizes and generate outputs:

```bash
.bundle/bin/python experiments_paper_inversion/run_experiment.py greedy_naive --mpi 10 --timeout 300
```

This will:
1. Run experiments for all configured sizes (from `sizes.yaml`)
2. Copy raw results to a timestamped output directory
3. Generate summary statistics
4. Generate LaTeX tables (supports multiple sizes)
5. Save all outputs in `experiments_paper_inversion/outputs/greedy_naive_YYYYMMDD_HHMMSS/`

### Running Multiple Sizes

```bash
.bundle/bin/python experiments_paper_inversion/run_all_sizes.py greedy_naive --mpi 10 --timeout 300
```

## Statistics Collected

**Per Replication:**
- Runtime with detailed breakdown (compute, solve, communication)
- True and estimated theta values for all parameters
- OLS and IV regression coefficients with standard errors
- Objective function values
- Complete metadata (replication ID, seed, dimensions, etc.)

**Summary Statistics:**
- Runtime: mean, std, median, min, max
- RMSE and Bias per parameter (with standard deviations)
- MAE and relative error percentages
- Timing breakdown averages
- Regression coefficient statistics with bias reduction metrics

**LaTeX Tables:**
- Professional formatting for top-tier journals (Econometrica)
- Automatic handling of multiple problem sizes in columns
- Runtime with error bars, RMSE/Bias per parameter
- Regression coefficients with standard errors
- Landscape mode for wide tables

## Notes

- Uses `.bundle` virtual environment (has statsmodels for IV experiments)
- Only row_generation solver (no 1-slack, no ellipsoid)
- Small debugging sizes configured (10-20 agents/items)
- Timeout protection prevents MPI hangs
- Tables automatically adapt to multiple sizes like experiments_paper


