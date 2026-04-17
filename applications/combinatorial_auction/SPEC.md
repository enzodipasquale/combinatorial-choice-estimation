# Combinatorial auction

Structural estimation of bidder demand from the FCC C-block auction (2000–2001),
following Fox and Bajari (2013). Each experiment ("spec") is a single YAML in
`configs/`; the same pipeline serves every spec and writes results under
`results/<spec>/`.

## Layout

```
data/
    datasets/           raw CSV inputs (do not edit)
    loaders.py          raw I/O, continental filter, context, aggregation matrix,
                        winner filtering, last-round capacity, Cholesky factor
    features.py         MODULAR / QUADRATIC / QUADRATIC_ID registries
    prepare.py          input_data + meta for combest

scripts/
    errors.py           error oracles (iid, feature-correlated, SAR,
                        pop/elig scaling, BTA→MTA aggregated oracle for CF)
    estimate.py         MPI entry: point estimation and bootstrap
    second_stage/       2SLS (iv.py), welfare decomposition (compute.py), tables
    counterfactual/     BTA→MTA aggregation, CF equilibrium solve
                        (with_xi / no_xi), bootstrap welfare, CF-vs-observed

configs/                <spec>.yaml — one per experiment
results/
    <spec>/
        point_estimate/     result.json
        bootstrap/          bootstrap_result.json, checkpoints/
        counterfactual/     cf_with_xi.json, cf_no_xi.json, bootstrap_welfare.json
run.sbatch              SLURM wrapper; takes config stem as argument
legacy/                 pre-rebuild code kept as a reference
legacy_results/         pre-rebuild estimation outputs (gitignored)
```

## Stages

1. **Estimate.** One MPI entry for point estimation and bootstrap, covering
   every error model (iid, feature-correlated, SAR, pop/elig scaled):
   ```
   mpirun -n N python -m applications.combinatorial_auction.scripts.estimate \
       applications/combinatorial_auction/configs/<spec>.yaml
   ```
   Writes `point_estimate/result.json` when `mode: estimation`, or
   `bootstrap/bootstrap_result.json` when `mode: bootstrap`.

2. **Second-stage tables.** 2SLS and welfare decomposition over bootstrap draws:
   ```
   python -m applications.combinatorial_auction.scripts.second_stage <spec1> <spec2> ...
   ```
   Reads `bootstrap/bootstrap_result.json`. Prints to stdout — no file saved.

3. **Counterfactual (point estimate).** α₀ and α₁ are obtained by running the
   2SLS block inline — never read from a saved file:
   ```
   mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.run <spec>
   ```
   Writes `counterfactual/cf_with_xi.json` and `counterfactual/cf_no_xi.json`.

4. **Counterfactual welfare (bootstrap).** Re-runs 2SLS and a CF solve per draw:
   ```
   mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.bootstrap_welfare <spec>
   ```
   Writes `counterfactual/bootstrap_welfare.json`.

5. **CF price vs observed.** Diagnostic printout:
   ```
   python -m applications.combinatorial_auction.scripts.counterfactual.analyze <spec> [--tag with_xi|no_xi]
   ```

## Spec config

Every spec YAML has an `application:` block. Keys:

```yaml
application:
    mode: bootstrap                 # 'estimation' | 'bootstrap'
    modular_regressors:             list
    quadratic_regressors:           list
    quadratic_id_regressors:        list (optional)
    winners_only:                   bool       (default false)
    capacity_source:                'initial' | 'last_round'   (default 'initial')
    error_seed:                     int
    error_correlation:              feature name or null
    spatial_rho:                    float or null   (SAR; mutually exclusive with error_correlation)
    error_scaling:                  'pop' | 'elig' | null

    counterfactual:                 # optional — drives the 2SLS consumed by the CF
        iv:                         'simple' | 'blp'            (default: blp if error_scaling=pop, else simple)
        regressors:                 list of column names        (default: preset)
        instruments:                list of column names        (default: preset)
        sample:                     'all' | 'rural'             (default: preset)
        pop_threshold:              int                         (default: 500_000)
        distance_threshold:         int (km)                    (default: 500)
```

Available column names (regressors / instruments):
`const`, `neg_price`, `pop`, `percapin`, `distant_pop_mean`, `distant_hhinc_mean`,
`blp_pop90`, `blp_percapin`, `blp_density`, `blp_hhinc35k`, `blp_grow9099`, `blp_imwl`.

The second-stage tables and the CF use the same `run_2sls()` call, so both
report the same α₀/α₁ for the same θ.

## Specs shipped

| spec | covariates | error model | default 2SLS |
|---|---|---|---|
| `boot` | elig_pop, adjacency, pop_centroid_delta4, elig_adjacency | iid | simple |
| `boot_3` | + air_travel, travel_survey, elig_pop_centroid_delta4 | iid | simple |
| `boot_5` | intermediate covariate set | iid | simple |
| `boot_pop_scaling` | same as boot | ε_ij · pop_j | BLP (rural) |
| `boot_pop_scaling_large` | + assets_pop, pop_centroid_00, elig_* | ε_ij · pop_j | BLP |
| `boot_pop_scaling_winners` | winners only | ε_ij · pop_j | BLP |
| `boot_pop_scaling_winners_item_only` | winners only, item-level quadratics only | ε_ij · pop_j | BLP |
| `boot_pop_scaling_winners_item_only_large` | above + extended covariates | ε_ij · pop_j | BLP |
| `spatial_correlated_rho0{0,2,4,6}` | same as boot | SAR(ρ) with ρ ∈ {0, .2, .4, .6} | simple |

## Conventions

- Prices and revenues in $B (BTA bid column divided by 1e9).
- `pop_j` and `elig_i` are both normalized by continental `pop_sum`; pop-scaled
  errors and the `elig_pop` feature therefore share the same scale.
- Continental filter (drop AK/HI/PR/Guam/etc.) is applied once in `data.loaders.load_raw`.
- CF offset: `mta_sizes · α₀ + A · Z'γ [ + A · ξ ]`; the `A · ξ` term toggles
  `with_xi` vs `no_xi`. Both variants are algebraically `A · δ + α₁ · A · p`.
- CF errors are drawn at BTA level, Cholesky-correlated / scaled as per config,
  aggregated through A, then offset-added — see `scripts.errors.install_aggregated`.

## Notes

- **Percapin as BLP IV.** The pre-rebuild `iv.py` on main had an uncommitted
  edit dropping `percapin` from the excluded instruments. The rebuild follows
  the working-directory behavior (no percapin). Add it back via
  `counterfactual.instruments` in the spec yaml if wanted.
- **FE convention in x̄.** `xbar` for the FE block is `b_obs.sum(axis=0)` rather
  than the strictly-consistent `-b_obs.sum(axis=0)`. With single-winner-per-BTA
  the two conventions are numerically identical within the welfare decomposition.
- **Cholesky PSD check.** `data.loaders.cholesky_factor` raises on non-PSD
  matrices — no silent fallback.
