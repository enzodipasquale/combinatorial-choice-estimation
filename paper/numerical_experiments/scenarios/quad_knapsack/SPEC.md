# Quadratic knapsack with BLP inversion — pilot SPEC

**Scenario 3 of the numerical experiments.** Showcases that `combest` scales to parameter vectors of size $M+2$ with a quadratic knapsack demand oracle, matching the size of the FCC combinatorial auction application ($N \approx 250$, $M \approx 500$).

**Scope of this pilot.** The agent generates the DGP, solves each agent's QKP exactly at $\theta^\star$ to produce observed bundles, and verifies the DGP is healthy. **No estimation is run in this pilot.** Row-generation estimation is a large HPC job launched separately once the DGP is validated.

## Economic framing

Agents $i = 1, \ldots, N$ choose bundles of items (e.g. spectrum licenses) $b_i \in \{0,1\}^M$ under a capacity constraint, with utility

$$V_i^\theta(b) = \alpha \sum_j x_{ij} b_j - \sum_j \delta_j b_j + \lambda \sum_{j < j'} Q_{j j'} b_j b_{j'} + \sum_j \nu_{ij} b_j$$

subject to $\sum_j w_j b_j \leq W_i$. Terms:

- $\alpha$ — modular coefficient on agent–item feature $x_{ij}$.
- $\delta_j$ — item fixed effect (BLP-style; $M$ nuisance parameters).
- $\lambda \geq 0$ — strength of pairwise complementarity; supermodular since $Q \geq 0$.
- $Q_{jj'} \in \{0, 1\}$ — **sparse** pairwise feature, geographic adjacency.
- $\nu_{ij}$ — modular errors, drawn i.i.d. at estimation time.

Total parameters: $K = M + 2$. Biggest parameter vector in the paper.

## BLP inversion

The $\delta_j$ are absorbed as fixed effects during row generation. Post-estimation,

$$\delta_j = \alpha_0 - \alpha_1 p_j + \phi_j^\top \beta + \xi_j$$

(where $p_j$ is an endogenous price, $\phi_j$ item characteristics, $\xi_j$ unobserved quality, $z_j$ excluded instruments correlated with $\phi_j$ but not $\xi_j$) is recovered by 2SLS on $\hat\delta$. This 2SLS step lives downstream; the pilot only needs to generate $(\delta^\star, p, \phi, z, \xi)$ in a mutually consistent way.

DGP recipe for $\delta^\star$:

1. Draw item characteristics $\phi_j \in \mathbb{R}^{K_\phi}$ ($K_\phi = 3$) and instruments $z_j \in \mathbb{R}^{K_\phi}$, correlated via $\phi_j = z_j + \rho \xi_j$ with $\rho = 0.5$, where $z_j, \xi_j \sim \mathcal{N}(0, 1)$ independently.
2. Set raw $\delta_j = \phi_j^\top \beta^\star + \xi_j$, then demean and rescale to std = 0.5. The rescaling implicitly rescales $\beta^\star$; save the effective $\beta^\star$ so the downstream 2SLS recovers it exactly.
3. Draw a "price" $p_j = \pi_0 + \pi_z^\top z_j + \pi_\xi \xi_j + u_j$ with $\pi_\xi > 0$ so 2SLS has a bite. The pilot doesn't exercise 2SLS beyond a smoke test.

**Agent responsibility in the pilot:** generate and save $\delta^\star, p, \phi, z, \xi, \beta^\star$ so a later 2SLS can recover $\beta^\star$ at machine precision when given $\hat\delta = \delta^\star$. Smoke test: run the 2SLS on $\delta^\star$ itself, check $\hat\beta = \beta^\star$ up to numerical tolerance.

## Sparse quadratic structure — critical design choice

**This determines whether the pilot succeeds.** We need exact optimization with **MIPGap = 0**. Gurobi can close the gap fast only if $Q$ is sparse and structured. Dense random $Q$ does not scale; the old supermodular scenario at $M=200$ struggled for this reason.

### Structure — geographic adjacency

Items placed on $[0,1]^2$ uniformly at random. For each item $j$, neighbors

$$\mathcal{N}(j) = \{j' \neq j : d(j, j') \leq r\},$$

and $Q_{jj'} = 1$ if $j' \in \mathcal{N}(j)$ else 0. Symmetric.

### Sparsity target

**Average degree $\bar k \approx 8$ independent of $M$.** Radius chosen so $\pi r^2 (M-1) \approx \bar k$, i.e. $r = \sqrt{\bar k / (\pi (M-1))}$.

At $M = 500, \bar k = 8$: radius $\approx 0.071$, nonzero pairs $\approx 2000$ (vs. $\sim 125\,000$ dense).

**Hard ceiling:** if realized nonzero count exceeds $15 M$, abort DGP generation. Gurobi will not scale.

### Why this shape

Matches the `adjacency` feature in `applications/combinatorial_auction`. Alternatives considered and rejected:

- **Block-diagonal / clusters:** factorizes trivially, misses the point — FCC quadratic is not block-diagonal.
- **Distance-decaying dense** ($Q_{jj'} = e^{-\alpha d}$): dense, too slow.
- **Random Bernoulli:** old scenario's choice; non-geographic, slow at $M=200$.

## Capacity constraint

Weights $w_j \sim U(0.5, 1.5)$. Capacities $W_i \sim U(0.3 \sum_j w_j, \ 0.5 \sum_j w_j)$ — agents pick roughly 30–50% of items by weight.

## Modular agent feature

$x_{ij} \sim \mathcal{N}(0, 1)$ i.i.d. Single coefficient $\alpha$.

## Parameters

$$\theta^\star = (\alpha^\star, \delta_1^\star, \ldots, \delta_M^\star, \lambda^\star) \in \mathbb{R}^{M+2}, \qquad \lambda^\star \geq 0.$$

Defaults (agent can tune during healthy search):

- $\alpha^\star = 0.1$
- $\delta^\star$: mean 0, std 0.5, constructed as above
- $\lambda^\star = 0.05$

## Sizes

**Pilot only.** Showcase is HPC.

| Size | $N$ | $M$ | Purpose |
|---|---|---|---|
| Tiny | 10 | 15 | Brute-force verification. DGP must match exhaustive enumeration exactly for $\geq 3$ agents. |
| Pilot | 30 | 50 | Healthy DGP validation. Full diagnostic suite. |
| Intermediate | 100 | 200 | Optional stress test of sparsity design if pilot is clean. |

**Not in scope for agent:** $N=250, M=500$ showcase.

## Healthy-DGP conditions

Agent runs these checks and aborts if any fail. Log all diagnostics either way.

1. **Optimization gap is zero.** Every per-agent QKP solves with `MIPGap = 0` within the per-agent time budget. If any agent hits the time limit with nonzero gap, DGP is unhealthy.

2. **Per-agent wall time.** Budget $T_\text{solve}$: 1s at $M=50$, 5s at $M=200$. Median $\ll$ budget; max $<$ budget.

3. **Bundle sparsity.** Mean bundle size in $[0.15 M, 0.45 M]$.

4. **Cross-agent heterogeneity.** Std of bundle sizes $\geq \max(5, 0.03 M)$.

5. **Item-level identification.** Every item appears in at least one agent's bundle AND is absent from at least one — for every $j$. Required for $\delta_j$ identification.

6. **Quadratic term binds.** Mean per-agent quadratic contribution $\lambda \sum_{j<j'} Q_{jj'} b_{ij}^\star b_{ij'}^\star$ across agents is $\geq 10\%$ of the mean modular contribution $|\sum_j (\alpha x_{ij} - \delta_j) b_{ij}^\star|$.

7. **Counterfactual-at-$\lambda=0$ differs.** Re-solve each agent with $\lambda = 0$. Mean Jaccard with $\lambda^\star$ solution $\leq 0.85$.

8. **Sparsity hit.** Realized average degree in $[4, 12]$. Total nonzeros $\leq 15 M$.

9. **Capacity binds.** For $\geq 80\%$ of agents, $\sum_j w_j b_{ij}^\star \in [0.95 W_i, W_i]$.

On failure, light random search over $\lambda^\star$ and rescaling of $\delta^\star$. Budget: 20 candidates.

## Solver requirements

- **`combest`'s `QuadraticKnapsackGRB` solver.** Do not roll your own.
- `MIPGap = 0` mandatory. `TimeLimit` per subproblem from size table above.
- **Any subproblem returning with positive gap = unhealthy DGP.** Do not accept near-optimal solutions.
- Brute-force verification at $M = 15$: exact match against exhaustive enumeration for $\geq 3$ random agents.

## Deliverable

`result.json` at pilot size ($N=30, M=50$):

```json
{
  "scenario": "quadratic_knapsack",
  "size": {"N": 30, "M": 50, "avg_degree": 7.8, "nnz_Q": 195},
  "theta_star": {
    "alpha": 0.1,
    "lambda": 0.05,
    "delta_summary": {"mean": 0.0, "std": 0.5, "min": ..., "max": ...}
  },
  "dgp_paths": {
    "phi": "data/phi.npy",
    "z": "data/z.npy",
    "xi": "data/xi.npy",
    "delta_star": "data/delta_star.npy",
    "beta_star": "data/beta_star.npy",
    "prices": "data/prices.npy",
    "Q": "data/Q_sparse.npz",
    "weights": "data/weights.npy",
    "capacities": "data/capacities.npy",
    "x_modular": "data/x_modular.npy",
    "obs_bundles": "data/obs_bundles.npy"
  },
  "healthy_checks": {
    "gurobi_max_gap": 0.0,
    "gurobi_max_wall_time_s": 0.42,
    "gurobi_median_wall_time_s": 0.11,
    "mean_bundle_size": 12.3,
    "std_bundle_size": 2.1,
    "item_identification_ok": true,
    "quad_contribution_fraction": 0.18,
    "bundle_jaccard_vs_lambda0": 0.62,
    "capacity_binding_fraction": 0.93,
    "all_checks_passed": true
  },
  "smoke_tests": {
    "brute_force_match_M15": true,
    "twosls_recovers_beta_star": true
  },
  "runtime_seconds": 34.2
}
```

Also a log file with per-agent solve times, gaps, bundle diagnostics.

## What the agent does NOT do

- No row-generation estimation.
- No bootstrap.
- No $M=500$ runs.
- No HPC job submission.
- No edits to `combest/` core.

The pilot validates that **exactly** this DGP can be generated and solved cleanly at $(N=30, M=50)$ with the sparsity and healthy conditions above. If it can, scaling to $(N=250, M=500)$ on HPC is compute, not design.
