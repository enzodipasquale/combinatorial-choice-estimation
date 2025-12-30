# Utility Function for Agent i Choosing Bundle B

## Gentzkow Two-Period Setting

For agent $i$ choosing bundle $B \subseteq \{1, \ldots, J\}$ (where $J = 2 \times J_{\text{period}}$ is the total number of items across 2 periods):

$$
U_i(B) = \underbrace{\sum_{j \in B} \sum_{k=1}^{K_{\text{mod,agent}}} X_{ijk} \theta_k^{\text{mod,agent}}}_{\text{Modular agent features}} 
+ \underbrace{\sum_{j \in B} \sum_{k=1}^{K_{\text{mod,item}}} Z_{jk} \theta_k^{\text{mod,item}}}_{\text{Modular item features}}
+ \underbrace{\sum_{j \in B} \sum_{\ell > j} \sum_{k=1}^{K_{\text{quad,item}}} Q_{j\ell k} \theta_k^{\text{quad,item}}}_{\text{Quadratic item features (block diagonal)}}
+ \underbrace{\sum_{j \in B} \varepsilon_{ij}}_{\text{Errors}}
$$

## Notation

- $X_{ijk}$: Modular agent feature $k$ for agent $i$ and item $j$ (shape: `(num_agents, num_items, num_mod_agent)`)
- $Z_{jk}$: Modular item feature $k$ for item $j$ (shape: `(num_items, num_mod_item)`)
- $Q_{j\ell k}$: Quadratic item feature $k$ for item pair $(j, \ell)$ (shape: `(num_items, num_items, num_quad_item)`)
  - **Block diagonal structure**: $Q_{j\ell k} = 0$ if items $j$ and $\ell$ are in different periods
  - Upper triangular: $Q_{j\ell k} = 0$ if $j \geq \ell$
  - Non-negative off-diagonals: $Q_{j\ell k} \geq 0$ for $j < \ell$ (supermodularity)
- $\varepsilon_{ij}$: Error term for agent $i$ and item $j$ (shape: `(num_agents, num_items)`)
- $\theta$: Parameter vector with components:
  - $\theta_k^{\text{mod,agent}}$: $k = 1, \ldots, K_{\text{mod,agent}}$ (modular agent parameters)
  - $\theta_k^{\text{mod,item}}$: $k = 1, \ldots, K_{\text{mod,item}}$ (modular item parameters)
  - $\theta_k^{\text{quad,item}}$: $k = 1, \ldots, K_{\text{quad,item}}$ (quadratic item parameters)

## Vectorized Form

Using bundle indicator vector $B \in \{0,1\}^J$:

$$
U_i(B) = \underbrace{B^T X_i \theta^{\text{mod,agent}}}_{\text{Modular agent}} 
+ \underbrace{B^T Z \theta^{\text{mod,item}}}_{\text{Modular item}}
+ \underbrace{\sum_{k=1}^{K_{\text{quad,item}}} \theta_k^{\text{quad,item}} \cdot B^T Q_{:,:,k} B}_{\text{Quadratic item}}
+ \underbrace{B^T \varepsilon_i}_{\text{Errors}}
$$

where:
- $X_i$ is the $(J \times K_{\text{mod,agent}})$ matrix of agent $i$'s modular features
- $Z$ is the $(J \times K_{\text{mod,item}})$ matrix of item modular features
- $Q_{:,:,k}$ is the $(J \times J)$ upper triangular matrix for quadratic feature $k$
- $\varepsilon_i$ is the $(J,)$ error vector for agent $i$

## Block Diagonal Structure

The quadratic features have block diagonal structure:
- Period 1: items $1, \ldots, J_{\text{period}}$ → $Q_{j\ell k} \neq 0$ only if $j, \ell \in \{1, \ldots, J_{\text{period}}\}$
- Period 2: items $J_{\text{period}}+1, \ldots, 2J_{\text{period}}$ → $Q_{j\ell k} \neq 0$ only if $j, \ell \in \{J_{\text{period}}+1, \ldots, 2J_{\text{period}}\}$
- Cross-period: $Q_{j\ell k} = 0$ if $j$ and $\ell$ are in different periods

## Example with Current Settings

With your current configuration:
- `num_mod_agent = 2` → $K_{\text{mod,agent}} = 2$
- `num_mod_item = 2` → $K_{\text{mod,item}} = 2$
- `num_quad_item = 2` → $K_{\text{quad,item}} = 2$
- `num_items_per_period = 25` → $J_{\text{period}} = 25$, $J = 50$

So the utility is:

$$
U_i(B) = \sum_{j \in B} \left[ X_{ij1} \theta_1^{\text{mod,agent}} + X_{ij2} \theta_2^{\text{mod,agent}} \right]
+ \sum_{j \in B} \left[ Z_{j1} \theta_1^{\text{mod,item}} + Z_{j2} \theta_2^{\text{mod,item}} \right]
+ \sum_{j \in B} \sum_{\ell > j} \left[ Q_{j\ell 1} \theta_1^{\text{quad,item}} + Q_{j\ell 2} \theta_2^{\text{quad,item}} \right] \cdot \mathbf{1}\{j, \ell \text{ in same period}\}
+ \sum_{j \in B} \varepsilon_{ij}
$$

where $\mathbf{1}\{j, \ell \text{ in same period}\}$ is the indicator that items $j$ and $\ell$ are in the same period (enforced by block diagonal structure).

