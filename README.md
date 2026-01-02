# BundleChoice

MPI-parallelized estimation for combinatorial discrete choice models using row generation (requires Gurobi).

## Install

```bash
git clone https://github.com/enzodipasquale/combinatorial-choice-estimation
cd combinatorial-choice-estimation
pip install -e .
```

Requires MPI (OpenMPI or MPICH). Gurobi is optional but needed for row generation.

## Quick start

```python
from bundlechoice import BundleChoice
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    agent_features = np.random.normal(0, 1, (100, 20, 5))
    errors = np.random.normal(0, 0.1, (1, 100, 20))
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5, "num_simulations": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50, "tolerance_optimality": 0.001}
})

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate choices
theta_true = np.ones(5)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Estimate
if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

theta_hat = bc.row_generation.solve()

if rank == 0:
    print(f"True: {theta_true}, Estimated: {theta_hat}")
```

Run with: `mpirun -n 10 python script.py`

## Data format

```python
input_data = {
    "agent_data": {
        "modular": np.array,      # (num_agents, num_items, num_features)
    },
    "item_data": {
        "modular": np.array,      # (num_items, num_features)
        "quadratic": np.array,    # (num_items, num_items, num_features) - optional
    },
    "errors": np.array,           # (num_simulations, num_agents, num_items)
    "obs_bundle": np.array,       # (num_agents, num_items)
    "constraint_mask": np.array   # (num_agents, num_items) - optional
}
```

## Subproblem solvers

- `Greedy` - Greedy selection
- `LinearKnapsack` - Linear utility with weight constraint
- `QuadKnapsack` - Quadratic utility with weight constraint (needs Gurobi)
- `QuadSupermodularNetwork` - Supermodular utilities via network flow
- `QuadSupermodularLovasz` - Supermodular utilities via Lovász extension
- `PlainSingleItem` - Single item choice

Set via config: `bc.load_config({"subproblem": {"name": "Greedy"}})`

## Custom features

```python
def my_features(agent_id, bundle, data):
    """Returns feature vector (num_features,)"""
    X = data["agent_data"]["modular"][agent_id]
    return X.T @ bundle

bc.features.set_oracle(my_features)
```

## Custom subproblem solver

```python
from bundlechoice.subproblems.base import SerialSubproblemBase

class MyOptimizer(SerialSubproblemBase):
    def initialize(self, agent_id):
        return {"capacity": self.local_data["agent_data"]["capacity"][agent_id]}
    
    def solve(self, agent_id, theta, problem_state):
        # Your optimization logic
        return bundle

bc.subproblems.load(MyOptimizer)
```

## Configuration

```yaml
dimensions:
  num_agents: 100
  num_items: 50
  num_features: 10
  num_simulations: 1

subproblem:
  name: "Greedy"

row_generation:
  max_iters: 100
  tolerance_optimality: 1e-6
  theta_ubs: 1000
  theta_lbs: -1000
```

Load with `bc.load_config("config.yaml")` or `bc.load_config(config_dict)`.

## Estimation methods

**Row generation** (requires Gurobi):
```python
theta = bc.row_generation.solve()
```

**Ellipsoid method** (no Gurobi):
```python
theta = bc.ellipsoid.solve()
```

## Requirements

- Python ≥3.9
- numpy ≥1.24
- scipy ≥1.10
- mpi4py ≥3.1
- gurobipy ≥11.0 (optional)
- networkx ≥3.0
- pyyaml ≥6.0

## Tests

```bash
mpirun -n 10 pytest bundlechoice/tests/ -k "not ellipsoid"
```

## Citation

```bibtex
@software{bundlechoice2025,
  author = {Di Pasquale, Enzo},
  title = {BundleChoice: Combinatorial Discrete Choice Estimation},
  year = {2025},
  url = {https://github.com/enzodipasquale/combinatorial-choice-estimation}
}
```

Enzo Di Pasquale • [ed2189@nyu.edu](mailto:ed2189@nyu.edu) • [@enzodipasquale](https://github.com/enzodipasquale)
