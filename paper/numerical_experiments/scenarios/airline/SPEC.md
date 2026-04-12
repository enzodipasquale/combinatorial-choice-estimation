# Airline / Gross Substitutes — Scenario Spec

## Economic setup

Each airline $i \in [N]$ chooses a subset of directed route to operate on a network of $K$ cities. The network is the full directed graph, so the item set is $E = \{(o,d) : o \neq d\}$ with $M = K(K-1)$ edges. Each airline has a set of hubs $\mathcal{H}_i \subseteq [K]$ drawn randomly (size 1–3, uniform over cities, no replacement).

The utility of operating route set $b \in \{0,1\}^M$ is

$$V_i^\theta(b) = \sum_{j \in E} b_j \phi_j^\top \theta^{\text{mod}} - \theta^{\text{gs}} \sum_{h \in \mathcal{H}_i} \Big( \sum_{j \in E : o(j) = h} b_j \Big)^2 + \varepsilon_i(b).$$

Modular revenue on each route net of cost, minus a convex congestion penalty at each hub (quadratic in the number of routes originating at that hub), plus idiosyncratic shocks.

**Gross substitutes.** For $\theta^{\text{gs}} \geq 0$, the deterministic part $V_i^\theta(b) - \varepsilon_i(b)$ is GS. With $\varepsilon_i(b) = \sum_j b_j \nu_{ij}$ (modular errors), the full valuation is also GS. Demand is computable by greedy.

## Parameters

All parameters are collected in $\theta$.

- $\theta^{\text{mod}} \in \mathbb{R}^{K_{\text{mod}}}$ — modular covariate coefficients.
- $\theta^{\text{gs}} \geq 0$ — congestion penalty.

No item fixed effects $\delta$. No BLP inversion.

## Covariates $\phi_j$

Two variants, selected by a config flag `fe_mode`:

**`fe_mode: "none"`.** $\phi_j$ has 2 components:
1. $\log(\text{origin\_pop}_j \cdot \text{dest\_pop}_j)$ — gravity-flavored revenue proxy.
2. $-\text{distance}_j$ — negative distance as a cost shifter.

**`fe_mode: "origin"`.** $\phi_j$ has $2 + K$ components: the two above plus one origin FE per city (one-hot on $o(j)$).

**Identification under origin FE.** For origin FEs to be identified, every city must be used as an origin by at least one but not all airlines in the observed data. The DGP-healthy-$\theta$ search (below) must enforce this.

## DGP

1. Draw city locations uniformly in $[0,1]^2$. Distances are Euclidean.
2. Draw city populations from a log-normal distribution (details tunable; use something with meaningful variation, e.g. $\log \text{pop} \sim \mathcal{N}(0, 0.5^2)$).
3. For each airline, draw hub set $\mathcal{H}_i$ uniformly at random, size in $\{1, 2, 3\}$.
4. Construct $\phi_j$ for each edge per the `fe_mode`.
5. Draw modular errors $\nu_{ij} \sim \mathcal{N}(0, \sigma^2)$ i.i.d. across $(i,j)$.
6. At the chosen true $\theta^*$, compute each airline's observed bundle $b_i^*$ via the greedy oracle.

## Healthy-DGP $\theta$ search

The true $\theta^*$ is **not hardcoded**. It is selected by a search procedure that ensures the generated bundles are informative.

Healthy-DGP criteria at $\theta^*$:

1. **Bundle size variation.** Mean bundle size in $(0.1 M, \; 0.5 M)$ — roughly between 10% and 50% of edges active. No airline chooses all routes; no airline chooses none.
2. **Cross-airline heterogeneity.** Standard deviation of bundle size across airlines $\geq$ some threshold (e.g., $0.05 M$). Not all airlines look identical.
3. **Under origin FE only:** every city has origin-utilization share in $(0, 1)$. Compute utilization$(o) = \frac{1}{N} \sum_i \mathbb{1}[\exists j : o(j)=o, b_{ij}^* = 1]$, require $\min_o \text{utilization}(o) > 0$ and $\max_o \text{utilization}(o) < 1$.

**Procedure.** Grid-search or simple random search over $\theta^{\text{gs}} \in [0.1, 5]$ and $\theta^{\text{mod}}$ components in reasonable ranges. For each candidate, run DGP once, check criteria, keep first healthy draw or best-scoring. Log rejected candidates and reasons.

## Oracle

Custom `find_best_item` for the greedy solver.

Given current bundle $b$ and candidate edge $j = (o,d)$, the marginal value of adding $j$ is:

$$\Delta_j(b) = \phi_j^\top \theta^{\text{mod}} + \nu_{ij} - \theta^{\text{gs}} \big[ (n_o + 1)^2 - n_o^2 \big] = \phi_j^\top \theta^{\text{mod}} + \nu_{ij} - \theta^{\text{gs}} (2 n_o + 1),$$

if $o \in \mathcal{H}_i$, where $n_o = |\{k \in E : o(k) = o, b_k = 1\}|$. If $o \notin \mathcal{H}_i$, the congestion term is zero and $\Delta_j(b) = \phi_j^\top \theta^{\text{mod}} + \nu_{ij}$.

`find_best_item` maintains the per-hub outbound counts $n_o$ and updates in $O(1)$ after each item added. Evaluating all remaining edges is $O(M)$. Total greedy cost: $O(M^2)$.

## Brute-force verification

At $M \leq 5$ (so $2^M \leq 32$), the greedy oracle must match brute-force enumeration: for 10 random airlines with random $\theta$ and random errors, greedy utility must equal brute-force utility (within $10^{-8}$). This check runs as part of the test suite.

## Estimation

The estimator is the regret-minimization estimator from the `combest` core. Parameters estimated: $\theta = (\theta^{\text{mod}}, \theta^{\text{gs}})$, with $\theta^{\text{gs}} \geq 0$ enforced via bounds.

Single simulation draw ($S = 1$). Single replication for the pilot.

## Deliverable

One JSON file `result.json` containing:

- `theta_true`: the $\theta^*$ found by the healthy-DGP search.
- `theta_hat`: the estimated $\theta$.
- `error_pct`: componentwise $(|\hat\theta_k - \theta^*_k| / |\theta^*_k|) \cdot 100$.
- `n_cities`, `n_edges` ($M$), `n_airlines` ($N$), `fe_mode`.
- `runtime_seconds`.
- `row_generation_iters`.
- `converged`: bool.
- `dgp_healthy_check`: dict with the three criteria and their observed values.

## Pilot size

$N = 30$ airlines, $M$ chosen so smoke tests run in seconds.
