# Paper Tables

This directory contains scripts to generate LaTeX tables for the paper.

## Generating the Parameter Estimates Table

```bash
cd paper_tables
../run-gurobi.bash python generate_table.py
```

## How It Works

The script reads from CSVs in `estimation_results/`:
- `theta_hat.csv` — parameter estimates (with delta metadata)
- `se_non_fe.csv` — standard errors (with delta metadata)

It filters by delta and fills the table with whatever data is available.

## Output

`parameter_estimates.tex` — LaTeX table with columns for both δ=2 and δ=4

## Table Structure

| Parameter | δ=4 Coef | δ=4 SE | δ=2 Coef | δ=2 SE |
|-----------|----------|--------|----------|--------|
| Bidder eligibility × population | ... | ... | ... | ... |
| Population/distance | ... | ... | ... | ... |
| Trips between markets | ... | ... | ... | ... |
| Total trips between airports | ... | ... | ... | ... |

Missing values show "---".
