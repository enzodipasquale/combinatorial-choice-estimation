# Multi-stage production facility location — numerical experiment

A `combest` demonstration on a multi-stage facility location problem inspired by HMMY (2026). Two parametrizations: a fully linear-in-$\theta$ version (implemented, estimated via row generation) and a convex-in-$\theta$ version (future work, requires DC outer loop).

## Setup

Each firm $i \in [N]$ chooses: cell plants to open, assembly plants to open, model–market entries. Bundle $b_i = (y^1_i, y^2_i, z_i)$ where

- $y^k_{i,g_k,\ell_k} \in \{0,1\}$: stage-$k$ facility activation at location $\ell_k \in L_k$ for production group $g_k \in G_k$ ($k=1$ cells, $k=2$ assemblies)
- $z_{i,m,n} \in \{0,1\}$: model $m$ entry in market $n$
- Path variables $x_{i,m,n,\ell_1,\ell_2} \in [0,1]$ route model $m$ serving market $n$ through cell $\ell_1$ and assembly $\ell_2$

Feasibility (identical to HMMY):

$$\sum_{\ell_1,\ell_2} x_{i,m,n,\ell_1,\ell_2} = z_{i,m,n}, \quad x_{i,m,n,\ell_1,\ell_2} \leq y^1_{i,\Gamma^1(m),\ell_1}, \quad x_{i,m,n,\ell_1,\ell_2} \leq y^2_{i,\Gamma^2(m),\ell_2}.$$

Locations drawn uniformly on the unit torus $\mathbb{T}^2$, with torus distance $d$. Each cell and assembly location is assigned a region $r(\ell_k) \in \{0, 1, 2\}$ by sorting x-coordinates into thirds.

## Linear-in-$\theta$ parametrization (implemented)

Structural utility:

$$V_i^\theta(b) = \sum_{m,n,\ell_1,\ell_2} \pi_{i,m,n,\ell_1,\ell_2}(\theta) \, x_{i,m,n,\ell_1,\ell_2} - \sum_{g_1,\ell_1} c^1_{i,g_1,\ell_1}(\theta) \, y^1_{i,g_1,\ell_1} - \sum_{g_2,\ell_2} c^2_{i,g_2,\ell_2}(\theta) \, y^2_{i,g_2,\ell_2}$$

with per-path revenue

$$\pi_{i,m,n,\ell_1,\ell_2}(\theta) = s_{i,m,n} R_n \big(1 - \rho^{d,1} d_{\ell_1,\ell_2} - \rho^{d,2} d_{\ell_2,n}\big),$$

and per-stage facility costs, with the region fixed effects moved to the **cost side** (linear) rather than the revenue side (convex in HMMY):

$$c^k_{i,g_k,\ell_k}(\theta) = \delta^k + \rho^{\xi,k} \tilde\xi_{i,g_k} + \rho^{HQ,k} D^{HQ}_{i,\ell_k} - \mathrm{FE}^k_{r(\ell_k)}.$$

Stochastic heterogeneity (modular, observed by firm):

$$\varepsilon_i(b) = \sum_{m,n,\ell_1,\ell_2} \nu_{i,m,n,\ell_1,\ell_2} x_{i,m,n,\ell_1,\ell_2} - \sum_{g_1,\ell_1} \tilde\phi^1_{i,g_1,\ell_1} y^1_{i,g_1,\ell_1} - \sum_{g_2,\ell_2} \tilde\phi^2_{i,g_2,\ell_2} y^2_{i,g_2,\ell_2}.$$

**Parameters ($K = 12$)** with the normalization $\mathrm{FE}^k_0 \equiv 0$:

$$\theta = \big(\delta^1, \delta^2,\ \rho^{\xi,1}, \rho^{\xi,2},\ \rho^{HQ,1}, \rho^{HQ,2},\ \mathrm{FE}^1_1, \mathrm{FE}^1_2,\ \mathrm{FE}^2_1, \mathrm{FE}^2_2,\ \rho^{d,1}, \rho^{d,2}\big).$$

$V_i^\theta(b) = \phi_i(b)^\top \theta + c_i(b)$ where $c_i(b) = \sum s_{i,m,n} R_n x_{i,m,n,\ell_1,\ell_2}$ is the $\theta$-independent revenue constant (absorbed into the error oracle). Row generation alone identifies $\theta$; the DC wrapper is vacuous.

## Convex-in-$\theta$ parametrization (future work)

Restore HMMY's CES revenue structure:

$$\pi_{i,m,n,\ell_1,\ell_2}(\theta) = s_{i,m,n} R_n \exp\Big((\eta - 1) \kappa \big[\mathrm{FE}^2_{r(\ell_2)} + \beta^\Phi \mathrm{FE}^1_{r(\ell_1)} - \rho^{d,1} d_{\ell_1,\ell_2} - \rho^{d,2} d_{\ell_2,n}\big]\Big).$$

Revenue is convex in $(\mathrm{FE}^1, \mathrm{FE}^2, \rho^{d,1}, \rho^{d,2})$ through the exponential. The criterion becomes difference-of-convex; estimation requires the DC outer loop linearizing $\pi$ around $\theta_k$ at each iterate. $K$ remains 12.

## Differences from HMMY

| Feature | HMMY | Ours |
|---|---|---|
| Continents / regions | 3 continents | 3 regions (from x-coord sort) |
| Geometry | Real lat/lon, Haversine | Random on $\mathbb{T}^2$, torus $d$ |
| Revenue in FE | Convex (CES, $\exp$) | Linear (FE on cost side) |
| Tariffs | Bilateral, calibrated | None |
| Sourcing productivity | Calibrated externally (nested DC) | Absorbed into $\rho^{d,k}$ |
| Market saturation | CES concavity in market quantity | None (linear in $x$) |
| $\delta^k_r$ per region | Yes ($3 \times 2 = 6$ params) | Scalar $\delta^k$ (2 params) |
| Fixed-cost FE sign | $\delta^k_{N(\ell_k)}$ additive | Subtracted ($-\mathrm{FE}^k_r$) as amenity |
| Quality $\tilde\xi$ | Log quality by chemistry/platform | $\tilde\xi \sim \mathcal{N}(0, 0.5)$ |
| HQ distance | Bilateral km | Torus $\log(1 + d)$ |
| Total parameters | 14 (4 FE + 6 $\delta$ + 2 $\rho^{HQ}$ + 2 $\rho^{\xi}$) | 12 |

The linear version trades HMMY's CES structure (convex revenue, concave sales) for linearity-in-$\theta$, making row generation directly applicable. The convex version recovers HMMY's revenue form and requires DC — our target for the methodological contribution.
