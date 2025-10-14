# BundleChoice

Estimate discrete choice models when agents pick bundles instead of single items. Firms don't just export to one country—they export to many. Bidders don't buy one spectrum license—they buy portfolios. This handles that.

The core problem: you observe bundle choices, you want to recover the utility parameters that rationalize them. Standard revealed preference estimation but for combinatorial choice sets.

## Why this is hard

The choice set explodes. With 50 items, there are 2^50 possible bundles. You can't enumerate them. So estimation requires:
- Solving an optimization problem per agent per iteration
- Checking rationality constraints you can't write down explicitly
- Distributing computation because it's slow

BundleChoice does row generation (fast, needs Gurobi) or ellipsoid method (slower, no Gurobi needed), parallelized with MPI.

## Install

```bash
git clone https://github.com/enzodipasquale/combinatorial-choice-estimation
cd combinatorial-choice-estimation
pip install -e .
```

You need MPI (OpenMPI or MPICH) and optionally Gurobi if you want row generation.

## Minimal example

```python
from bundlechoice import BundleChoice
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Data on rank 0
if rank == 0:
    agent_features = np.random.normal(0, 1, (100, 20, 5))  # 100 agents, 20 items, 5 features
    errors = np.random.normal(0, 0.1, (1, 100, 20))
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

# Setup
bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5, "num_simuls": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50, "tolerance_optimality": 0.001}
})

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate choices with true parameters
theta_true = np.ones(5)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Add observations and estimate
if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

theta_hat = bc.row_generation.solve()

if rank == 0:
    print(f"True: {theta_true}")
    print(f"Estimated: {theta_hat}")
```

Run it:
```bash
mpirun -n 10 python script.py
```

## How it works

BundleChoice has five components:

**DataManager** - Scatters data from rank 0 across MPI processes. Each rank gets a slice of agents.

**FeatureManager** - Computes utility features from bundles. Either auto-generated from your data structure or custom function you provide.

**SubproblemManager** - Solves each agent's utility maximization. Picks from built-in algorithms or uses your custom solver.

**RowGenerationSolver** - Main estimation loop. Gurobi LP for master problem + your subproblem solver as separation oracle.

**EllipsoidSolver** - Alternative estimation that doesn't need Gurobi. Slower but sometimes useful.

## Built-in subproblem solvers

| Name | What it does |
|------|--------------|
| `Greedy` | Greedily add items one by one |
| `LinearKnapsack` | Linear utility + weight constraint |
| `QuadKnapsack` | Quadratic utility + weight constraint (uses Gurobi) |
| `QuadSupermodularNetwork` | Supermodular utilities via network flow |
| `QuadSupermodularLovasz` | Supermodular utilities via Lovász extension |
| `PlainSingleItem` | Just picks one item (standard discrete choice) |

Pick one:
```python
bc.load_config({"subproblem": {"name": "QuadSupermodularNetwork"}})
```

## Data structure

```python
input_data = {
    "agent_data": {
        "modular": np.array,      # (num_agents, num_items, num_features)
    },
    "item_data": {
        "modular": np.array,      # (num_items, num_features)
        "quadratic": np.array,    # (num_items, num_items, num_features) - optional
    },
    "errors": np.array,           # (num_simuls, num_agents, num_items)
    "obs_bundle": np.array,       # (num_agents, num_items)
    "constraint_mask": np.array   # (num_agents, num_items) - optional, boolean
}
```

The `modular` arrays get used to auto-generate features. If you have quadratic terms, add them under `item_data["quadratic"]`.

## Custom features

Auto-generated features work for simple cases but you'll often want custom logic:

```python
def my_features(agent_id, bundle, data):
    """
    agent_id: index on this MPI rank (int)
    bundle: binary array (num_items,)
    data: local data dictionary
    
    Returns: feature vector (num_features,)
    """
    X = data["agent_data"]["modular"][agent_id]  # (num_items, num_features)
    return X.T @ bundle

bc.features.set_oracle(my_features)
```

The feature function gets called a lot, so make it fast. The greedy solver will automatically detect if your oracle can handle vectorized computation (multiple bundles at once) and use that if available.

## Custom subproblem solver

If the built-in solvers don't fit your problem:

```python
from bundlechoice.subproblems.base import SerialSubproblemBase
import numpy as np

class MyOptimizer(SerialSubproblemBase):
    def initialize(self, agent_id):
        """Setup for one agent. Return whatever state you need."""
        capacity = self.local_data["agent_data"]["capacity"][agent_id]
        return {"capacity": capacity}
    
    def solve(self, agent_id, theta, problem_state):
        """Solve for this agent given parameters theta."""
        # Compute utilities
        utilities = []
        for j in range(self.num_items):
            bundle = np.zeros(self.num_items)
            bundle[j] = 1
            features = self.features_oracle(agent_id, bundle)
            utilities.append(theta @ features)
        
        # Your optimization logic
        bundle = np.zeros(self.num_items)
        bundle[np.argmax(utilities)] = 1
        return bundle

bc.subproblems.load(MyOptimizer)
```

Inherit from `SerialSubproblemBase` if you solve per agent, or `BatchSubproblemBase` if you can vectorize across agents.

## Configuration

Settings go in a dict or YAML file:

```yaml
dimensions:
  num_agents: 100
  num_items: 50
  num_features: 10
  num_simuls: 1

subproblem:
  name: "Greedy"
  settings: {}

row_generation:
  max_iters: 100
  tolerance_optimality: 1e-6
  theta_ubs: 1000
  theta_lbs: -1000
  gurobi_settings:
    Method: 0        # Primal simplex
    OutputFlag: 0    # Quiet
```

Load it:
```python
bc.load_config("config.yaml")
# or
bc.load_config(config_dict)
```

You can update config incrementally:
```python
bc.load_config({"row_generation": {"max_iters": 200}})  # Merges with existing
```

## Typical workflow

```python
bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()  # or set_oracle(fn)
bc.subproblems.load()
theta = bc.row_generation.solve()
```

Or use the shortcut:
```python
bc = BundleChoice().quick_setup(config, input_data, features_oracle=None)
theta = bc.row_generation.solve()
```

## Row generation details

The row generation solver iterates:
1. Master problem (Gurobi LP): find θ satisfying current rationality constraints
2. Separation oracle: for each agent, solve utility maximization with current θ
3. Add violated constraints: if any agent could improve, add that constraint

Stops when converged or max iterations hit.

Config options:
- `tolerance_optimality`: convergence threshold (default 1e-6)
- `max_iters`: hard stop (default infinity)
- `min_iters`: force at least this many iterations (default 0)
- `theta_ubs` / `theta_lbs`: bounds on parameters (defaults: 1000, None)
- `gurobi_settings`: pass through to Gurobi model

Typical convergence is 20-50 iterations for synthetic data.

## Ellipsoid method

Alternative that doesn't need Gurobi:

```python
theta = bc.ellipsoid.solve()
```

Uses subgradient cuts to shrink an ellipsoid around the feasible region. Slower than row generation but doesn't require a commercial solver.

Config:
```python
bc.load_config({
    "ellipsoid": {
        "max_iterations": 1000,
        "tolerance": 1e-6,
        "initial_radius": 1.0
    }
})
```

## Advanced stuff

**Constraint masks** - exclude items for specific agents:
```python
# Firms can't export to their home country
constraint_mask = np.ones((num_agents, num_items), dtype=bool)
constraint_mask[i, home_country[i]] = False
input_data["constraint_mask"] = constraint_mask
```

**Quadratic utilities** - for complementarities:
```python
input_data["item_data"]["quadratic"] = ...  # (num_items, num_items, num_features)
bc.load_config({"subproblem": {"name": "QuadSupermodularNetwork"}})
```

**Callbacks** - monitor progress:
```python
def callback(info):
    print(f"Iter {info['iteration']}: theta={info['theta']}")

theta = bc.row_generation.solve(callback=callback)
```

The callback gets a dict with: `iteration`, `theta`, `objective`, `max_violation`, and timing info.

## Debugging

Check what's initialized:
```python
bc.print_status()
```

Output tells you what's missing. Common issues:

- **"Cannot initialize subproblem manager"** - you didn't load data or set features yet
- **Gurobi license error** - use `bc.ellipsoid.solve()` instead
- **Wrong results** - check your feature function returns shape `(num_features,)`
- **Slow** - try a simpler subproblem solver

Validate before running:
```python
bc.validate_setup('row_generation')  # Raises informative error if something's wrong
```

## Tests

```bash
# All tests except ellipsoid (slow)
mpirun -n 10 pytest bundlechoice/tests/ -k "not ellipsoid"

# Specific solver
mpirun -n 10 pytest bundlechoice/tests/test_greedy.py
```

## Project layout

```
bundlechoice/
├── core.py                    # Main BundleChoice class
├── data_manager.py            # MPI data distribution
├── feature_manager.py         # Feature computation
├── config.py                  # Configuration objects
│
├── subproblems/
│   ├── base.py                # Base classes
│   ├── subproblem_manager.py
│   └── registry/              # Built-in solvers
│       ├── greedy.py
│       ├── linear_knapsack.py
│       ├── quadratic_knapsack.py
│       └── quad_supermodular/
│
├── estimation/
│   ├── row_generation.py
│   ├── ellipsoid.py
│   └── inequalities.py
│
└── tests/
```

## Requirements

- Python ≥3.9
- numpy ≥1.24
- scipy ≥1.10
- mpi4py ≥3.1
- gurobipy ≥11.0 (optional, for row generation)
- networkx ≥3.0 (for supermodular solvers)
- pyyaml ≥6.0

## License

MIT

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
