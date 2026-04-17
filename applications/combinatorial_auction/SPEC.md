# Combinatorial auction

Structural estimation of bidder demand from the FCC C-block auction (Fox and
Bajari, 2013). All stages key off the spec name (= config stem = results
subdirectory).

## Layout

```
data/
    datasets/           raw CSVs
    loaders.py          raw I/O, continental filter, context, aggregation matrix
    features.py         MODULAR / QUADRATIC / QUADRATIC_ID registries
    prepare.py          c-block input_data + meta

scripts/
    errors.py           error-oracle factory (iid, correlated, SAR, pop/elig)
    estimate.py         MPI entry: point estimation + bootstrap
    second_stage/       2SLS, welfare decomposition, tables
    counterfactual/     BTA→MTA prepare, CF solve, bootstrap welfare, analyze

configs/                <spec>.yaml
results/                <spec>/point_estimate/, bootstrap/, counterfactual/
run.sbatch              SLURM wrapper
legacy/                 pre-rebuild code
```

## Pipeline

```
# 1. Estimate (point or bootstrap, per config mode)
mpirun -n N python -m applications.combinatorial_auction.scripts.estimate configs/<spec>.yaml

# 2. Welfare decomposition across bootstrap draws
python -m applications.combinatorial_auction.scripts.second_stage <spec> [<spec> ...]

# 3. Counterfactual on the point estimate (α₀, α₁ auto-derived via 2SLS)
mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.run <spec>

# 4. Per-draw CF welfare (BTA vs MTA)
mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.bootstrap_welfare <spec>

# diagnostic
python -m applications.combinatorial_auction.scripts.counterfactual.analyze <spec> [--tag with_xi|no_xi]
```

## Configs

| spec | covariates / notes | errors | 2SLS |
|---|---|---|---|
| `boot` | elig_pop + adjacency + pop_centroid_delta4 + elig_adjacency | iid | simple |
| `boot_3` | + air_travel, travel_survey, elig_pop_centroid_delta4 | iid | simple |
| `boot_5` | intermediate set | iid | simple |
| `boot_pop_scaling` | same as boot | ε · pop | BLP (rural) |
| `boot_pop_scaling_large` | + assets_pop, pop_centroid_00, elig_* | ε · pop | BLP |
| `boot_pop_scaling_winners*` | winners-only sample, item-only variants | ε · pop | BLP |
| `spatial_correlated_rho0{0,2,4,6}` | same as boot | SAR ρ ∈ {0,.2,.4,.6} | simple |

## Conventions

- Prices in $B (bid / 1e9).
- `pop` and `elig` normalized by continental `pop_sum`; pop-scaled errors and
  the `elig_pop` feature share this scale.
- CF offset: `mta_sizes · α₀ + A · (Z'γ) [ + A · ξ ]` (the ξ term toggles
  `with_xi` vs `no_xi`; algebraically `A·δ + α₁·A·p` either way).
- `counterfactual.iv` (simple|blp) and its regressors/instruments/sample/
  thresholds are overridable in the spec YAML; defaults come from `error_scaling`.
