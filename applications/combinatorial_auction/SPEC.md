# Combinatorial Auction (C-Block)

Structural estimation of demand and welfare from the FCC C-block spectrum auction (2000-01), following Fox and Bajari (2013).

## Structure

```
data/
    loaders.py          Raw data loading, context construction, aggregation matrices
    prepare.py          Feature construction for estimation (modular, quadratic, quadratic_id)
    registries.py       Feature registry with decorator-based registration
    errors.py           Error covariance (Cholesky factor for SAR/correlated errors)
    iv.py               IV regression utilities (OLS, 2SLS, BLP instruments)
    analysis/           Descriptive plots and maps
    datasets/           Raw CSV data (Fox-Bajari replication archive)

scripts/
    estimate.py         Main entry point: point estimation and bootstrap (MPI)

    c_block/
        configs/            YAML experiment configurations
        second_stage/    Second-stage analysis (runs after estimation)
            compute.py              Per-draw IV + surplus decomposition from bootstrap results
            tables.py               LaTeX-style output tables
            bootstrap_welfare.py    Counterfactual welfare comparison per bootstrap draw (MPI)
            __main__.py             Entry: python -m ...second_stage [specs]
        counterfactual/     MTA-level counterfactual equilibrium
            prepare.py              Aggregate BTA estimation to MTA structure
            run.py                  Solve counterfactual with/without unobservables
            analyze.py              Compare counterfactual to observed outcomes
            errors.py               Counterfactual error construction (MTA-level)
        sar_robustness/     Spatially-autoregressive error robustness checks
        zero_noise/         Deterministic error variants (elig_pop fixed at 1)
        extract_bootstrap_uhat.py   HPC utility: parse Gurobi checkpoints
```

## Pipeline

1. **Estimate**: `mpirun -n N python scripts/estimate.py configs/boot.yaml`
2. **Post-estimation tables**: `python -m scripts.c_block.second_stage boot boot_3`
3. **Counterfactual**: `mpirun -n N python scripts/c_block/counterfactual/run.py configs/cf_boot.yaml`
4. **Bootstrap welfare**: `mpirun -n N python scripts/c_block/second_stage/bootstrap_welfare.py results/boot/bootstrap_result.json`

## Preferred specifications

- `boot`: baseline (elig_pop + adjacency + pop_centroid_delta4 + elig_adjacency), iid errors
- `boot_3`: extended quadratics (+ air_travel, travel_survey, elig_pop_centroid_delta4)
- `boot_pop_scaling`: baseline covariates, pop-scaled errors (pop_j * epsilon_ij)
- `boot_pop_scaling_large`: pop-scaled + additional covariates (assets_pop, pop_centroid_00, etc.)
