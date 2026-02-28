# Numerical Experiments Pipeline

This pipeline runs Monte Carlo experiments to fill Table 1 (benchmarks table) from the paper.

## Structure

- `generate_data.py`: Data generation with endogeneity (modular inversion setting)
- `run_experiment.py`: Run a single Monte Carlo replication
- `compute_statistics.py`: Compute bias, RMSE, SE across replications
- `aggregate_results.py`: Aggregate results into table format
- `run_all_experiments.py`: Run all experiments
- `config.yaml`: Configuration file

## Quick Start

### Debug Mode (Small Sizes)

Test with N=10, M=10:

```bash
cd paper/numerical_experiments
python run_all_experiments.py --debug
```

### Run Single Replication

```bash
mpirun -n 10 python run_experiment.py --spec gross_substitutes --N 10 --M 10 --replication 0
```

### Run All Experiments

```bash
python run_all_experiments.py
```

This will:
1. Run 50 replications for each (specification, M, N) combination
2. Compute statistics for each combination
3. Generate `results/table.csv` and `results/table.tex`

## Configuration

Edit `config.yaml` to adjust:
- Number of replications (`experiment.n_replications`)
- Bootstrap samples (`experiment.n_bootstrap`)
- Lambda values for calibration
- Grid of M and N values
- Maximum M for quadratic knapsack (default: 50)

## Specifications

1. **Gross Substitutes**: Uses `Greedy` subproblem
2. **Supermodular**: Uses `QuadraticSupermodularMinCut` subproblem
3. **Linear Knapsack**: Uses `LinearKnapsackGRB` subproblem
4. **Quadratic Knapsack**: Uses `QuadraticKnapsackGRB` subproblem (stops at M=50 by default)

## Lambda Calibration

Lambda values are set in `config.yaml`. For gross substitutes and supermodular specifications, adjust `lambda_gs` and `lambda_quad` to ensure bundles are well-posed (mean bundle size between 30-70% of M).

To calibrate lambda for a given M:
1. Run a test replication with different lambda values
2. Check bundle sizes: `mean(bundle_sizes) in [0.3*M, 0.7*M]`
3. Update `config.yaml` with calibrated values

## Output

- `results/raw/`: Individual replication results (JSON)
- `results/stats_*.json`: Statistics per (spec, N, M) combination
- `results/table.csv`: Final table in CSV format
- `results/table.tex`: Final table in LaTeX format

## Notes

- Uses MPI with 10 processes (configurable via `--mpi-procs`)
- Timeout wrapper prevents hanging (60s for M=50, 120s for M=100, 240s for M=200)
- Bootstrap uses Bayesian bootstrap with 200 resamples
- Runtime excludes bootstrap time (point estimation only)
