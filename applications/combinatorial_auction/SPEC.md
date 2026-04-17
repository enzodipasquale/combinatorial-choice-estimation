# Combinatorial auction

Structural estimation of bidder demand from the FCC C-block auction (2000–2001),
following Fox and Bajari (2013). Each spec runs through the same pipeline;
results are written under `results/<spec>/`.

## Layout

```
data/
    datasets/           raw CSV inputs (do not edit)
    loaders.py          raw I/O, continental filter, context, aggregation matrix,
                        winner filtering, last-round capacity, Cholesky factor
    features.py         MODULAR / QUADRATIC / QUADRATIC_ID registries
    prepare.py          c-block input_data + meta for combest

pipeline/
    errors.py           error-oracle factory (iid, feature-correlated, SAR,
                        pop/elig scaling, BTA→MTA aggregated oracle for CF)
    estimate.py         MPI entry for point estimation + bootstrap
    second_stage/       2SLS, welfare decomposition, pretty tables
    counterfactual/     BTA→MTA aggregation, CF equilibrium solve (with_xi and
                        no_xi), bootstrap welfare, CF-vs-observed analysis

configs/                <spec>.yaml — one per experiment
results/                <spec>/ with result.json, bootstrap_result.json,
                        cf_{with_xi,no_xi}.json, bootstrap_welfare.json
run.sbatch              SLURM wrapper; takes config stem as argument
legacy/                 pre-rebuild code, kept for comparison only
```

## Pipeline

All stages key off the spec name (= config file stem = results subdirectory).

1. **Estimate.** Single MPI entry for point estimation and bootstrap, for any
   error model (iid, feature-correlated, SAR, pop/elig scaled):
   ```
   mpirun -n N python -m applications.combinatorial_auction.pipeline.estimate \
       applications/combinatorial_auction/configs/<spec>.yaml
   ```
   Writes `result.json` (when `mode: estimation`) or `bootstrap_result.json`
   (when `mode: bootstrap`).

2. **Second stage.** Welfare decomposition tables across bootstrap draws:
   ```
   python -m applications.combinatorial_auction.pipeline.second_stage <spec1> <spec2> ...
   ```

3. **Counterfactual (point estimate).** α₀ and α₁ auto-derived from 2SLS
   (BLP IV if `error_scaling: pop`, otherwise simple IV):
   ```
   mpirun -n N python -m applications.combinatorial_auction.pipeline.counterfactual.run <spec>
   ```
   Writes `cf_with_xi.json` and `cf_no_xi.json`.

4. **Counterfactual welfare (bootstrap).** Re-runs 2SLS + CF per draw:
   ```
   mpirun -n N python -m applications.combinatorial_auction.pipeline.counterfactual.bootstrap_welfare <spec>
   ```

5. **CF price vs observed.** Diagnostic printout:
   ```
   python -m applications.combinatorial_auction.pipeline.counterfactual.analyze <spec> [--tag with_xi|no_xi]
   ```

## Configs

| spec | covariates | error model | second stage |
|---|---|---|---|
| `boot` | elig_pop, adjacency, pop_centroid_delta4, elig_adjacency | iid | simple IV |
| `boot_3` | + air_travel, travel_survey, elig_pop_centroid_delta4 | iid | simple IV |
| `boot_5` | intermediate covariate set | iid | simple IV |
| `boot_pop_scaling` | same as boot | ε_ij × pop_j | BLP IV (rural) |
| `boot_pop_scaling_large` | + assets_pop, pop_centroid_00, elig_* | ε_ij × pop_j | BLP IV |
| `boot_pop_scaling_winners` | winners-only bidders | ε_ij × pop_j | BLP IV |
| `boot_pop_scaling_winners_item_only` | winners-only + `item_modular: price` | ε_ij × pop_j | BLP IV |
| `boot_pop_scaling_winners_item_only_large` | above + extended covariates | ε_ij × pop_j | BLP IV |
| `spatial_correlated_rho0{0,2,4,6}` | same as boot | SAR with ρ ∈ {0, .2, .4, .6} | simple IV |

## Conventions

- Prices and revenues in $B (BTA bid column divided by 1e9).
- `pop_j` and `elig_i` are both normalized by continental `pop_sum`, so
  pop-scaled errors and the `elig_pop` feature share the same scale.
- Continental filter (drop AK/HI/PR/Guam/etc.) applied once in `loaders.load_raw`.
- CF offset: `mta_sizes · α₀ + A · (Z'γ)  [ + A · ξ ]` where the `A · ξ` term
  toggles the `with_xi` vs `no_xi` variants (algebraically `A·δ + α₁·A·p`
  either way — invariant to how δ splits between α₀, Z'γ, and ξ).
- CF errors: drawn at BTA level, Cholesky-correlated/scaled, aggregated via A,
  then offset added — implemented by `pipeline.errors.install_aggregated`.

## Known differences from the pre-rebuild pipeline

- `sar_robustness/run.py` (165 LOC) collapsed into a config-driven variant of
  `pipeline.estimate`; SAR covariance is built by `pipeline.errors.build_sar_correlation`.
- `bootstrap_welfare.py` no longer rebuilds features from the registry; it
  reuses `counterfactual.run.solve_cf`.
- `iv.py` moved from `data/` to `pipeline/second_stage/`.
- BLP IV matches the current runnable `data/iv.py` on main (without `percapin`
  in the excluded instruments — see note below).

## Notes / to confirm

- **Percapin as BLP IV.** Main's committed `iv.py` includes `percapin` in the
  excluded instruments; main's working directory has an uncommitted edit that
  drops it. The rebuild follows the working-directory version (drops percapin).
- **FE convention in x̄.** Following the current pipeline, `xbar` for the FE
  block is `b_obs.sum(axis=0)` rather than the strictly-consistent
  `-b_obs.sum(axis=0)`. With single-winner-per-BTA this is numerically
  equivalent to `fe_total = δ.sum()`; confirm before publication.
- **Cholesky PSD check.** `loaders.cholesky_factor` raises on non-PSD matrices
  (matches combest behavior); there is no silent fallback.
