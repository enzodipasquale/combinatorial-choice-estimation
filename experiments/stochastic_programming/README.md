# Dynamic Bundle Choice Problem

This folder contains scripts for implementing and testing a dynamic bundle choice problem with additions and removals.

## Problem Formulation

The problem solves:

\[
\max_{\{d_t,e_t\}} \mathbb{E}\!\left[\sum_{t=1}^T \beta^t f(b_t, d_t, e_t, \eta_t)\right]
\]

Subject to:
- State transition: \(b_{t+1} = b_t + d_t - e_t\)
- Initial state: \(b_0 \in \{0,1\}^J\) given
- Feasibility: \(e_t \le b_t\), \(d_t \le \mathbf{1} - b_t\)
- Decisions are \(\mathcal{F}_t\)-measurable

The objective function is:
\[
\begin{aligned}
f(b,d,e,\eta) 
&= \sum_{j} b_j (r_j - c_j)
\;+\; \sum_{j<i} b_j b_i \,(c_j + c_i)\, S_{ji} \\[4pt]
&\quad - \sum_j d_j F_j
\;+\; \sum_{j<i} d_j d_i \,(F_j + F_i)\, S_{ji} \\[4pt]
&\quad + \sum_j e_j r_j
\;+\; \sum_j \eta_j .
\end{aligned}
\]

## Current Implementation

The script `dynamic_bundle_problem.py` implements a deterministic approximation where:
- Stochastic shocks \(\eta_t\) are averaged over Monte Carlo scenarios
- Decision variables \(d_t, e_t\) are the same across all scenarios

This is simpler but not fully stochastic. For true stochastic programming, we would need:
- Scenario-specific decision variables for future periods
- Non-anticipativity constraints ensuring decisions at time \(t\) only depend on information revealed up to time \(t\)

## Usage

```python
from dynamic_bundle_problem import DynamicBundleProblem, generate_test_data

# Generate test data
data = generate_test_data(
    num_actual_items=5,
    num_periods=3,
    num_mc_scenarios=20,
    seed=42
)

# Create and solve problem
problem = DynamicBundleProblem(
    num_actual_items=5,
    num_periods=3,
    beta=0.95,
    num_mc_scenarios=20
)

problem.set_data(**data)
d_optimal, e_optimal, obj_value = problem.solve()
```

## Bundle Choice Convention

In bundle choice, `num_items = num_actual_items * num_periods`. Items are ordered as:
```
[items_period_1, items_period_2, ..., items_period_T]
```

This allows the problem to be integrated into the bundle choice framework where each "item" represents an actual item in a specific time period.





















