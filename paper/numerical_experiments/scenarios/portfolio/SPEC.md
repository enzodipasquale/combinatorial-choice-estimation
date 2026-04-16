# Scenario: Continuous Controls in Combest — Portfolio Choice DGP

## Purpose

Test whether `combest`'s zero-noise estimator can recover parameters of a portfolio-style problem in which the agent's choice combines a binary inclusion vector $s \in \{0,1\}^M$ and a continuous weight vector $w \in \mathbb{R}_+^M$, with $w_j > 0 \implies s_j = 1$.

The framework currently assumes binary `obs_bundles`. This scenario probes whether a minimal extension — storing both $s$ and $w$ as a single float array of shape $(N, 2M)$ — works end-to-end without restructuring the framework.

## Encoding convention

Each observation is stored as a float vector of length $2M$:
- `b[:, :M]` = inclusion vector $s$
- `b[:, M:]` = weight vector $w$

Consistency $w_j > 0 \implies s_j = 1$ is enforced by the subproblem solver and by construction of the DGP. The framework does not check it.

## Framework edit

A single line in `combest/subproblems/subproblem_manager.py`, method `generate_obs_bundles`:

```python
# BEFORE
self.data_manager.local_data.id_data["obs_bundles"] = local_bundles.astype(bool)
# AFTER
self.data_manager.local_data.id_data["obs_bundles"] = local_bundles
```

This is the only modification to `combest/`. No other framework code needs to change. Existing applications must continue to pass tests after this edit.

## DGP

**Dimensions:** $N = 200$ agents, $M = 15$ items, $K = 3$ characteristics.

**Items:** characteristic matrix $X \in \mathbb{R}^{M \times K}$ with entries drawn $\mathcal{N}(0, 1)$. Per-agent shifters $X_i = X + \eta_i$ with $\eta_i \sim \mathcal{N}(0, 0.1^2)$ to generate cross-agent variation in optimal portfolios.

**Covariance:** $\Sigma = \Lambda \Lambda^\top + \sigma_\xi^2 I_M$ with $\Lambda \in \mathbb{R}^{M \times 2}$ drawn $\mathcal{N}(0, 1)$ and $\sigma_\xi = 0.3$. Symmetric positive definite by construction. Treated as known.

**True parameters:** $\beta^* = (1.0,\, 0.5,\, -0.3)$, $\gamma^* = 2.0$, $\kappa^* = 0.05$.

**Agent's problem:**
$$\max_{s \in \{0,1\}^M,\; w \in \mathbb{R}_+^M} \quad \sum_{j=1}^M w_j \, (X_{ij}^\top \beta) \;-\; \frac{\gamma}{2}\, w^\top \Sigma w \;-\; \kappa \sum_{j=1}^M s_j$$
subject to $\sum_j w_j = 1$ and $w_j \leq s_j$ for all $j$.

No idiosyncratic errors. Zero-noise regime.

## Subproblem solver

A new class inheriting from `combest.subproblems.subproblem_solver.SubproblemSolver`. Solves the agent's MIQP via Gurobi, one model per agent. The `solve(theta)` method returns an array of shape `(n_local, 2*M)` as float, with the first $M$ entries the binary $s$ (cast to float) and the last $M$ entries the weights $w$.

**Theta ordering:** `theta = [beta_0, beta_1, beta_2, gamma, kappa]` (length 5).

Use the FCC C-block solver (`applications/combinatorial_auction/`) as a structural template for class scaffolding (init, MPI conventions, agent-loop pattern). The optimization itself differs.

## Covariates oracle

Signature: `covariates_oracle(bundles, ids, data) -> ndarray of shape (n, 5)`.

Given `bundles` of shape `(n, 2M)`:
```python
s = bundles[:, :M]
w = bundles[:, M:]
```

Returns features as columns, in theta order:
- $\phi_{\beta_k}(s, w) = \sum_j w_j \, X_{ij,k}$ for $k = 0, 1, 2$
- $\phi_\gamma(s, w) = -\tfrac{1}{2}\, w^\top \Sigma\, w$
- $\phi_\kappa(s, w) = -\sum_j s_j$ (use `s.sum(-1)`, never `bundles.sum(-1)`)

Per-agent characteristics $X_i$ and the shared $\Sigma$ are retrieved from `data.id_data` and `data.item_data` respectively; the oracle should be a closure over these or read them from `data`.

**Error oracle:** returns zeros. Zero-noise regime.

## Estimation

In `run.py`:

1. Build DGP. Generate $\{(s_i^*, w_i^*)\}_{i=1}^N$ by calling the subproblem solver at $\theta^*$.
2. Pack into `obs_bundles` of shape $(N, 2M)$ as float. Pass via `load_and_distribute_input_data` (do not use `generate_obs_bundles`, which would invoke the framework's MPI scatter path and is unnecessary here).
3. Configure `row_generation.theta_bounds`: pin `beta_0 = 1.0` via `lbs[0] = ubs[0] = 1.0` for zero-noise normalization. The remaining four parameters are free.
4. Set the covariates and error oracles via `model.features.set_covariates_oracle(...)` and `model.features.set_error_oracle(...)`.
5. Run `model.point_estimation.n_slack.solve(verbose=True)`.

## Diagnostics to report

- Recovered ratios $\hat\beta_1/\hat\beta_0,\, \hat\beta_2/\hat\beta_0,\, \hat\gamma/\hat\beta_0,\, \hat\kappa/\hat\beta_0$ versus truth (with $\hat\beta_0 = 1$ fixed).
- Per-agent $|s_i^*|$ distribution (mean, std, min, max) — sanity check that the DGP produces non-degenerate sparsity. If all agents pick $|s| = 1$ or $|s| = M$, parameters are mistuned.
- Number of row-generation iterations to convergence.
- Walltime.
- Final objective value.

## Acceptance criterion

The script runs to completion. Estimated ratios are within 5% of true ratios. The existing test suite (`mpirun -n 10 pytest combest/tests/`) continues to pass after the framework edit.

If recovery fails, do not silently retune the DGP. Report what was attempted and the magnitude of the gap.

## Constraints on execution

- Local only. No SLURM, no HPC.
- Activate venv before any run: `source combest/.bundle/bin/activate`.
- Run with `mpirun -n 4 python run.py`.
- Do not modify `combest/` beyond the single line in `subproblem_manager.py`.
- Do not modify the existing FCC or firms_export applications.
- All new code lives under the scenario directory.

## Files to create

- `dgp.py` — synthetic data generation (returns `X_i`, `Sigma`, `theta_star`)
- `solver.py` — MIQP subproblem solver class
- `oracle.py` — covariates and error oracles
- `run.py` — end-to-end driver: DGP → solve at $\theta^*$ → estimate → print diagnostics
- `README.md` — what the experiment does, how to run, expected outputs
