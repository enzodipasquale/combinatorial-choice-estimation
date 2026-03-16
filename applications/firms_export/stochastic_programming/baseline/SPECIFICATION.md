# Utility Specification — Baseline Two-Stage Model

## Setup

Firm $i$ has **initial export state** $b_0 \in \{0,1\}^M$ (destinations
currently served). It chooses:

- **Period 1**: bundle $b_1 \in \{0,1\}^M$
- **Period 2**: continuation bundles $b_2^{(r)} \in \{0,1\}^M$ for each
  scenario $r = 1, \dots, R$

## Parameters

- $\theta = (\theta_{\text{rev}},\; \theta_s,\; \theta_{sd},\; \theta_{\text{syn}})$
- Discount factor $\beta \in [0,1)$

## Data

- $x^{(t)}_{ij}$: revenue characteristics for firm $i$, destination $j$,
  period $t$ (`rev_chars_1`, `rev_chars_2`; dimension $n_{\text{rev}} \times M$
  per firm)
- $d_j$: entry cost characteristic for destination $j$ (`entry_chars`; e.g.
  distance)
- $C$: symmetric $M \times M$ synergy matrix (`syn_chars`), with zero diagonal

## Period 1 payoff

$$
\pi_1(b_1) = \sum_j b_{1j} \Big[
  \theta_{\text{rev}}' x^{(1)}_{ij}
  + (1 - b_{0j})\big(\theta_s + \theta_{sd}\, d_j\big)
  + (1 - b_{0j})\,\theta_{\text{syn}}\,(b_0' C)_j\, d_j
  + \varepsilon^{(1)}_{ij}
\Big]
$$

where the entry/synergy terms only apply to new destinations
($1 - b_{0j} = 1$).

## Period 2 payoff (scenario $r$)

$$
\pi_2^{(r)}(b_2^{(r)} \mid b_1) = \sum_j b_{2j}^{(r)} \Big[
  \tfrac{\beta}{1-\beta}\,\theta_{\text{rev}}' x^{(2)}_{ij}
  + \varepsilon^{(2,r)}_{ij}
\Big]
+ \sum_j (1-b_{1j})\,b_{2j}^{(r)}\Big[
  \beta\big(\theta_s + \theta_{sd}\, d_j\big)
  + \beta\,\theta_{\text{syn}}\,(b_1' C)_j\, d_j
\Big]
$$

Entry into destination $j$ in period 2 requires $j \notin b_1$ (i.e.
$1 - b_{1j} = 1$).

## Total value (two-stage stochastic program)

$$
V_i(\theta) = \max_{b_1} \left[
  \pi_1(b_1)
  + \frac{1}{R}\sum_{r=1}^R \max_{b_2^{(r)}} \pi_2^{(r)}(b_2^{(r)} \mid b_1)
\right]
$$

## Discounting

| Quantity | Discount factor | Rationale |
|---|---|---|
| Period-2 revenue | $\frac{\beta}{1-\beta}$ | Geometric perpetuity from period 2 onward |
| Period-2 entry cost | $\beta$ | One-shot cost upon switching |
| Period-2 synergy | $\beta$ | One-shot cost upon switching |

These are pre-computed by `build_oracles` and stored in data:

- `rev_chars_2_d` $= \frac{\beta}{1-\beta}\, x^{(2)}$
- `entry_chars_2` $= \beta\, d$
- `C_d_2` $= \beta\, C \odot d$  (element-wise: $\beta\, C_{jk}\, d_j$)
- `discount_2` $= \beta$  (scalar, for the entry intercept $\theta_s$)

The solver never references $\beta$ directly.

## Error structure

Errors are **independent across periods**:

- $\varepsilon^{(1)}_{ij} \sim N(0, \sigma_1^2)$ — period-1 shock
- $\varepsilon^{(2,r)}_{ij} \sim N(0, \sigma_2^2)$ — period-2 shock, i.i.d.
  across scenarios $r$

Drawn per firm using deterministic seeding: `(seed, firm_id, 0)` for period 1,
`(seed, firm_id, 1)` for period 2.

## Parameter interpretation

| Parameter | Covariate | Sign |
|---|---|---|
| $\theta_{\text{rev}}$ | Revenue characteristics | $+$ means higher revenue $\Rightarrow$ more likely to export |
| $\theta_s$ | Entry cost intercept (per new destination) | $-$ expected (cost) |
| $\theta_{sd}$ | Entry cost $\times$ distance | $-$ expected (farther $\Rightarrow$ costlier) |
| $\theta_{\text{syn}}$ | Synergy: overlap with current network $\times$ distance | $+$ means network effects reduce entry cost |

## Special cases

- **Static model** ($\beta = 0$, $R = 1$): period-2 terms vanish, reduces to
  a single-period combinatorial choice with quadratic synergy. Equivalent to
  the `QuadraticSupermodularMinCutSolver` formulation.
