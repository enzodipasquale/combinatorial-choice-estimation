# Quadratic knapsack with BLP inversion — pilot SPEC

**Scenario 3 of the numerical experiments.** Showcases that `combest` scales to parameter vectors of size $M+2$ with a quadratic knapsack demand oracle, matching the size of the FCC combinatorial auction application ($N \approx 250$, $M \approx 500$).

**Scope of this pilot.** Generate the DGP, solve each agent's QKP exactly at $\theta^\star$ to produce observed bundles, and verify the DGP is healthy. **No estimation is run in this pilot.** Row-generation estimation is a separate HPC job.

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
