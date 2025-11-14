# Inversion Experiments for Paper

This directory contains scripts for running numerical experiments comparing naive (biased) vs IV (unbiased) estimation methods with endogeneity.

## Structure

Top-level organization highlights the most important folders:
- `00_logs/` – Centralized SLURM outputs (`*.out`, `*.err`)
- `01_outputs/` – Timestamped experiment outputs created by the pipeline
- `02_estimations/` – All estimation code grouped by method
  - `naive/<experiment>/`
  - `iv/<experiment>/`
  (Legacy names such as `greedy_naive` are still auto-detected by the scripts if present.)

If a combined directory like `greedy/` is provided, the pipeline will use it directly; otherwise it pairs the naive and IV folders from `02_estimations/`.

Each directory contains:
- `config.yaml` - Experiment configuration (optional)
- `sizes.yaml` - Problem size configurations (3x3 grid: N=[5,10,20] x M=[10,20,30])
- `run.py` - Main experiment script
- `results.csv` - Results from running experiments (appended on each replication)

## Running Experiments

### Quick Start: Complete Pipeline

Use the main pipeline script to run both naive and IV methods:

```bash
python experiments_paper_inversion/run_experiment.py greedy --mpi 10 --timeout 600
```

This will:
1. Run experiments for all configured sizes (from `sizes.yaml`) for both naive and IV
2. Combine results from both methods into single CSV
3. Copy raw results to a timestamped output directory
4. Generate summary statistics
5. Generate LaTeX tables (3x3 grid format comparing naive vs IV)
6. Save all outputs in `experiments_paper_inversion/01_outputs/greedy_YYYYMMDD_HHMMSS/`

### Running Multiple Sizes

```bash
python experiments_paper_inversion/run_all_sizes.py experiments/naive/greedy --mpi 10 --timeout 600
```

Or for a specific experiment directory:
```bash
python experiments_paper_inversion/run_all_sizes.py experiments/iv/greedy --mpi 10 --timeout 600
```

### Generating Tables Only

To generate LaTeX tables from existing results:

```bash
python experiments_paper_inversion/run_experiment.py greedy --skip-run
```

## Output Format

### Results CSV (`results.csv`)

Each row contains:
- **Metadata**: `replication`, `seed`, `method` (naive/iv), `time_s`, `obj_value`, `num_agents`, `num_items`, `num_features`, `num_simuls`, `sigma`, `subproblem`
- **Timing breakdown**: `timing_compute`, `timing_solve`, `timing_comm`, and their percentages
- **True parameters**: `theta_true_0`, `theta_true_1`, ..., `theta_true_{K-1}`
- **Estimated parameters**: `theta_0`, `theta_1`, ..., `theta_{K-1}`
- **Regression results**: `ols_coef_0`, `ols_se_0`, `iv_coef_0`, `iv_se_0` (for IV method)

### Summary CSV (`results_summary.csv`)

Contains aggregated statistics per method (naive vs IV):
- `runtime_mean`, `runtime_std`: Average runtime over replications
- `rmse_0`, `rmse_1`, ..., `rmse_{K-1}`: Root Mean Squared Error per parameter
- `bias_0`, `bias_1`, ..., `bias_{K-1}`: Bias (mean error) per parameter
- Regression coefficient statistics with bias reduction metrics

### LaTeX Tables (`tables.tex`)

Generated automatically with 3x3 grid format:
- Shows Runtime, RMSE, and Bias for each parameter type
- Compares Naive (Biased) vs IV (Unbiased) methods side-by-side
- 3x3 grid of sizes: N=[5,10,20] x M=[10,20,30]
- Portrait-friendly format (no landscape needed)
- Professional formatting suitable for top-tier venues

To generate tables:
```bash
python experiments_paper_inversion/generate_latex_tables.py results.csv --experiment greedy --output tables.tex
```

### Timestamped Outputs

The `run_experiment.py` script creates organized outputs in `experiments_paper_inversion/01_outputs/experiment_YYYYMMDD_HHMMSS/`:
- `results_raw.csv`: Combined results from both naive and IV methods
- `results_summary.csv`: Aggregated statistics
- `tables.tex`: LaTeX tables (3x3 grid format)
- `config.yaml`: Configuration used
- `sizes.yaml`: Size definitions
- `README.md`: Metadata and information

## Notes

- Both naive and IV methods run automatically in single experiment
- Results are combined into single CSV for easy comparison
- Uses `.bundle` virtual environment (has statsmodels for IV experiments)
- Only row_generation solver (no 1-slack, no ellipsoid)
- 3x3 grid sizes configured: N=[5,10,20] x M=[10,20,30] (9 size combinations)
- Timeout protection prevents MPI hangs (default: 600s = 10 minutes)
- Tables automatically adapt to available sizes and show "---" for missing combinations


