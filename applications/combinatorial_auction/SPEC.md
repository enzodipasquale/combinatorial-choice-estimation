# Combinatorial Auction (C-Block)

Structural estimation of demand and welfare from the FCC C-block spectrum auction (2000-01), following Fox and Bajari (2013).

## Structure

```
data/
    loaders.py          Raw data loading, context construction, aggregation matrices
    prepare.py          Feature construction for estimation (modular, quadratic, quadratic_id)
    registries.py       Feature registry with decorator-based registration
    iv.py               IV regression utilities (OLS, 2SLS, BLP instruments)
    analysis/           Descriptive plots and maps
    datasets/           Raw CSV data (Fox-Bajari replication archive)

scripts/
    estimate.py         Main entry point: point estimation and bootstrap (MPI)

    c_block/
        configs/            YAML experiment configurations
        results/            Estimation and counterfactual outputs
        second_stage/       Second-stage analysis from bootstrap results
            compute.py          Per-draw IV + surplus decomposition
            tables.py           Output tables
        counterfactual/     MTA-level counterfactual
            prepare.py          Aggregate BTA estimation to MTA structure
            run.py              Solve counterfactual equilibrium
            analyze.py          Compare counterfactual to observed
            bootstrap_welfare.py  Welfare comparison per bootstrap draw (MPI)
        sar_robustness/     SAR error robustness checks
        zero_noise/         Deterministic error variants
        extract_bootstrap_uhat.py   HPC utility: parse Gurobi checkpoints
```

## Pipeline

1. **Estimate**: `mpirun -n N python scripts/estimate.py c_block/configs/boot.yaml`
2. **Second-stage tables**: `python -m scripts.c_block.second_stage boot boot_3`
3. **Counterfactual**: `mpirun -n N python scripts/c_block/counterfactual/run.py c_block/configs/cf_boot.yaml`
4. **Bootstrap welfare**: `mpirun -n N python scripts/c_block/counterfactual/bootstrap_welfare.py results/boot/bootstrap_result.json`
