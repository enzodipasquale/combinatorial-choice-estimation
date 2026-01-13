# BundleChoice

MPI-parallel estimation for combinatorial discrete choice models via row generation.

## Install

```bash
pip install -e .
```

Requires MPI. Gurobi optional but needed for row generation.

## Quick Start

```python
from bundlechoice import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary
from mpi4py import MPI

# Build scenario
scenario = (
    ScenarioLibrary.greedy()
    .with_dimensions(num_agents=200, num_items=30)
    .with_num_features(5)
    .build()
)

# Prepare and estimate
prepared = scenario.prepare(comm=MPI.COMM_WORLD, seed=42)
bc = BundleChoice()
prepared.apply(bc, comm=MPI.COMM_WORLD, stage="estimation")

result = bc.row_generation.solve()
print(result.theta_hat)
```

Run: `mpirun -n 10 python script.py`

## Scenarios

```python
ScenarioLibrary.greedy()                 # Greedy selection
ScenarioLibrary.linear_knapsack()        # Linear utility + capacity
ScenarioLibrary.quadratic_knapsack()     # Quadratic utility + capacity
ScenarioLibrary.quadratic_supermodular() # Supermodular utilities
ScenarioLibrary.plain_single_item()      # Single item choice
```

## Custom Data

```python
bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5},
    "subproblem": {"name": "Greedy"},
})

if rank == 0:
    input_data = {
        "agent_data": {"modular": features},  # (agents, items, features)
        "errors": errors,                      # (sims, agents, items)
        "obs_bundle": bundles,                 # (agents, items)
    }
else:
    input_data = None

bc.data.load_and_scatter(input_data)
bc.oracles.build_from_data()
result = bc.row_generation.solve()
```

## Standard Errors

```python
se_result = bc.standard_errors.compute_bayesian_bootstrap(result.theta_hat)
```

## Tests

```bash
mpirun -n 10 pytest bundlechoice/tests/
```
