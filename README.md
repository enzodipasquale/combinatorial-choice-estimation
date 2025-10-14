# BundleChoice

A Python toolkit for estimating discrete choice models when agents select bundles of items rather than single alternatives. Handles the case where you observe choices and want to recover preference parameters—think firms choosing export destinations, bidders selecting spectrum licenses, or consumers picking product portfolios.

## The problem

Standard discrete choice works when people pick one thing. But many decisions involve selecting multiple items simultaneously. The challenge: given observed bundle choices, estimate the underlying utility parameters that rationalize those choices.

This is hard because:
- The choice set grows exponentially with the number of items
- You need to solve an optimization problem for each agent at each iteration
- Estimation requires checking rationality constraints across all possible bundles

BundleChoice solves this using row generation (fast, needs Gurobi) or ellipsoid methods (slower, Gurobi-free), with MPI parallelization across agents.

## Installation

```bash
git clone https://github.com/enzodipasquale/combinatorial-choice-estimation
cd combinatorial-choice-estimation
pip install -e .
```

**Requirements:**
- Python ≥3.9
- MPI installation (OpenMPI or MPICH)
- Gurobi license (optional, only for row generation)

## Quick start

```python
from bundlechoice import BundleChoice
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Generate data on rank 0
if rank == 0:
    num_agents = 100
    num_items = 20
    num_features = 5
    
    agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
    errors = np.random.normal(0, 0.1, (1, num_agents, num_items))
    
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

# Initialize and configure
bc = BundleChoice()
bc.load_config({
    "dimensions": {
        "num_agents": 100,
        "num_items": 20,
        "num_features": 5,
        "num_simuls": 1
    },
    "subproblem": {"name": "Greedy"},
    "row_generation": {
        "max_iters": 50,
        "tolerance_optimality": 0.001
    }
})

# Load and scatter data across MPI ranks
bc.data.load_and_scatter(input_data)

# Auto-generate feature oracle from data structure
bc.features.build_from_data()

# Generate observed bundles (simulate choices)
theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Add observations to data
if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# Estimate parameters
theta_hat = bc.row_generation.solve()

if rank == 0:
    print(f"True:      {theta_true}")
    print(f"Estimated: {theta_hat}")
```

Run with:
```bash
mpirun -n 10 python script.py
```

## Architecture

BundleChoice follows a modular design with five core components:

### 1. DataManager
Handles MPI data distribution and local data access.

```python
bc.data.load_and_scatter(input_data)  # Scatter data from rank 0
local_data = bc.data.local_data        # Access local slice
```

**Data structure:**
```python
input_data = {
    "agent_data": {
        "modular": np.array,      # Shape: (num_agents, num_items, num_features)
        # ... other agent-specific data
    },
    "item_data": {
        "modular": np.array,      # Shape: (num_items, num_features)
        "quadratic": np.array,    # Shape: (num_items, num_items, num_features)
    },
    "errors": np.array,           # Shape: (num_simuls, num_agents, num_items)
    "obs_bundle": np.array        # Shape: (num_agents, num_items)
}
```

### 2. FeatureManager
Computes utility features from bundles.

**Auto-generated oracle:**
```python
bc.features.build_from_data()
```

This automatically creates a feature function that sums modular features over selected items.

**Custom oracle:**
```python
def my_features(agent_id, bundle, data):
    """
    Args:
        agent_id: Index of agent (local to this rank)
        bundle: Binary array of shape (num_items,)
        data: Dictionary with local data slices
    
    Returns:
        Feature vector of shape (num_features,)
    """
    X = data["agent_data"]["modular"][agent_id]  # (num_items, num_features)
    return X.T @ bundle  # (num_features,)

bc.features.set_oracle(my_features)
```

**Vectorized computation:**
```python
# Compute features for multiple bundles at once
bundles = np.array([[1,0,1,0], [0,1,1,0]])  # 2 bundles, 4 items
features = bc.features.compute_vectorized(agent_id, bundles)
# Returns: (2, num_features)
```

### 3. SubproblemManager
Solves the agent's utility maximization problem.

**Built-in solvers:**

| Solver | Use case | Method |
|--------|----------|--------|
| `Greedy` | Unconstrained, myopic selection | Greedy insertion |
| `GreedyOptimized` | Same, faster | Vectorized greedy |
| `GreedyJIT` | Same, fastest | Numba JIT compilation |
| `LinearKnapsack` | Linear utility + capacity | Dynamic programming |
| `QuadraticKnapsack` | Quadratic utility + capacity | MIP (Gurobi) |
| `QuadSupermodularNetwork` | Supermodular with quadratic | Network flow |
| `QuadSupermodularLovasz` | Same | Lovász extension |
| `PlainSingleItem` | Single item choice | Argmax |

**Usage:**
```python
bc.load_config({"subproblem": {"name": "Greedy"}})
bc.subproblems.load()

# Solve for all agents with given parameters
bundles = bc.subproblems.init_and_solve(theta)
# Returns: (num_agents, num_items) on rank 0, local slice on other ranks
```

**Custom solver:**
```python
from bundlechoice.subproblems.base import SerialSubproblemBase
import numpy as np

class MyOptimizer(SerialSubproblemBase):
    def initialize(self, agent_id):
        """Set up problem for one agent."""
        # Return problem state (optional)
        return {"weights": self.local_data["item_data"]["weights"]}
    
    def solve(self, agent_id, theta, problem_state):
        """Solve for one agent given theta."""
        # Compute utilities
        utilities = np.zeros(self.num_items)
        for j in range(self.num_items):
            bundle = np.zeros(self.num_items)
            bundle[j] = 1
            features = self.features_oracle(agent_id, bundle)
            utilities[j] = theta @ features
        
        # Your optimization logic here
        best_bundle = np.zeros(self.num_items)
        best_bundle[np.argmax(utilities)] = 1
        return best_bundle

# Load custom solver
bc.subproblems.load(MyOptimizer)
```

### 4. Row Generation Solver
Main estimation algorithm using Gurobi LP + separation oracle.

```python
theta_hat = bc.row_generation.solve()
```

**How it works:**
1. **Master problem**: LP that finds parameters satisfying rationality constraints
2. **Separation oracle**: For each agent, solve utility maximization with current θ
3. **Add violated constraints**: If any agent could improve, add that constraint
4. **Iterate**: Until convergence or max iterations

**Configuration:**
```python
bc.load_config({
    "row_generation": {
        "max_iters": 100,              # Max iterations
        "min_iters": 10,               # Force at least this many
        "tolerance_optimality": 1e-6,  # Convergence tolerance
        "theta_ubs": 10.0,             # Upper bound on parameters
        "theta_lbs": -10.0,            # Lower bound on parameters
        "gurobi_settings": {
            "Method": 0,               # Primal simplex
            "OutputFlag": 0,           # Suppress output
            "LPWarmStart": 2           # Use warm start
        }
    }
})
```

**Warm start:**
```python
# Provide initial theta
bc.row_generation_manager = RowGenerationSolver(
    ...,
    theta_init=theta_initial
)
```

### 5. Ellipsoid Solver
Alternative estimation without Gurobi (slower but more flexible).

```python
theta_hat = bc.ellipsoid.solve()
```

**Configuration:**
```python
bc.load_config({
    "ellipsoid": {
        "max_iterations": 1000,
        "tolerance": 1e-6,
        "initial_radius": 1.0,
        "decay_factor": 0.95,
        "verbose": True
    }
})
```

## Configuration

Settings can be provided as a dictionary or YAML file:

```yaml
dimensions:
  num_agents: 100
  num_items: 50
  num_features: 10
  num_simuls: 1

subproblem:
  name: "QuadSupermodularNetwork"
  settings:
    # Solver-specific options

row_generation:
  max_iters: 100
  tolerance_optimality: 0.001
  theta_ubs: 10.0
  gurobi_settings:
    Method: 0
    OutputFlag: 0

ellipsoid:
  max_iterations: 1000
  tolerance: 1e-6
```

Load configuration:
```python
bc.load_config("config.yaml")
# or
bc.load_config(config_dict)
```

**Update configuration:**
```python
# Merge new settings with existing
bc.load_config({"row_generation": {"max_iters": 200}})
```

## API patterns

### Standard workflow
```python
bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()  # or set_oracle(fn)
bc.subproblems.load()
theta_hat = bc.row_generation.solve()
```

### Quick setup
```python
bc = BundleChoice().quick_setup(config, input_data, features_oracle=None)
theta = bc.row_generation.solve()
```

### Generate observations
```python
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.generate_observations(theta_true)  # Automatically reloads data
theta_hat = bc.row_generation.solve()
```

### Temporary config
```python
with bc.temp_config(row_generation={'max_iters': 5}):
    quick_theta = bc.row_generation.solve()
# Config restored after context
```

### Status checking
```python
bc.print_status()
# Output:
# === BundleChoice Status ===
# Config:      ✓
# Data:        ✓
# Features:    ✓
# Subproblems: ✓
# 
# Dimensions:  agents=100, items=20, features=5
# Algorithm:   Greedy
# MPI:         rank 0/10

# Or programmatic check
status = bc.status()
if not status['features_set']:
    bc.features.build_from_data()
```

### Validation
```python
bc.validate_setup('row_generation')  # Raises error if incomplete
bc.validate_setup('ellipsoid')
```

## Advanced features

### Constraint masks

Exclude certain items for specific agents (e.g., firms can't export to home country):

```python
# Boolean mask: True = feasible, False = excluded
constraint_mask = np.ones((num_agents, num_items), dtype=bool)
for i in range(num_agents):
    home_country = home_countries[i]
    constraint_mask[i, home_country] = False

input_data["constraint_mask"] = constraint_mask
bc.data.load_and_scatter(input_data)
```

The solver automatically respects these constraints during optimization.

### Quadratic utilities

For complementarities between items:

```python
input_data = {
    "agent_data": {"modular": ...},
    "item_data": {
        "modular": ...,
        "quadratic": ...  # (num_items, num_items, num_features)
    }
}

bc.load_config({"subproblem": {"name": "QuadSupermodularNetwork"}})
```

Utility: `θ' f(bundle)` where features include both linear and quadratic terms.

### Multiple simulations

Run multiple simulations with different error draws:

```python
errors = np.random.normal(0, 1, (num_simuls, num_agents, num_items))
input_data["errors"] = errors

bc.load_config({"dimensions": {"num_simuls": 10}})
```

### Callbacks

Monitor convergence:

```python
def progress_callback(info):
    print(f"Iter {info['iteration']}: obj={info['objective']:.4f}, "
          f"violation={info['max_violation']:.6f}")

theta = bc.row_generation.solve(callback=progress_callback)
```

## Testing

Run the test suite:

```bash
# All tests (excluding slow ellipsoid tests)
mpirun -n 10 pytest bundlechoice/tests/ -k "not ellipsoid"

# Specific solver
mpirun -n 10 pytest bundlechoice/tests/test_greedy.py

# With coverage
mpirun -n 10 pytest bundlechoice/tests/ --cov=bundlechoice
```

Tests cover:
- All built-in solvers
- Row generation and ellipsoid estimation
- Data distribution across MPI ranks
- Feature computation (auto and custom)
- Configuration loading and validation
- Edge cases and error handling

## Debugging

Common issues and solutions:

**"Cannot initialize subproblem manager"**
```python
bc.print_status()
# Check what's missing, then:
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
```

**Gurobi license error**
```python
# Use ellipsoid method instead
theta = bc.ellipsoid.solve()
```

**Slow convergence**
```python
# Increase tolerance or decrease max_iters
bc.load_config({
    "row_generation": {
        "tolerance_optimality": 0.01,  # Looser tolerance
        "max_iters": 50
    }
})
```

**MPI errors**
```bash
# Always run with mpirun
mpirun -n 10 python script.py
```

**Wrong results**
```python
# Validate your feature oracle
features = bc.features.features_oracle(agent_id=0, bundle=np.array([1,0,1,0]))
print(features)  # Should have shape (num_features,)

# Check observed bundles are being used
print(bc.data.local_data["obs_bundle"])
```

## Performance

Typical timings (100 agents, 50 items, 5 features):

| Solver | Time per iteration | Convergence |
|--------|-------------------|-------------|
| Greedy | ~0.5s | 20-40 iters |
| GreedyJIT | ~0.1s | 20-40 iters |
| LinearKnapsack | ~2s | 30-50 iters |
| QuadSupermodular | ~5s | 40-60 iters |

**MPI scaling:**
- Linear speedup up to ~20 processes
- Beyond that, communication overhead dominates
- Best practice: Use `num_processes ≤ num_agents / 5`

**Memory usage:**
- Rank 0: Stores full data + Gurobi model
- Other ranks: Only their data slice
- Largest arrays: features matrix, observed bundles

## Project structure

```
bundlechoice/
├── __init__.py
├── core.py                    # BundleChoice main class
├── base.py                    # Base classes and mixins
├── config.py                  # Configuration dataclasses
├── data_manager.py            # MPI data distribution
├── feature_manager.py         # Feature computation
├── comm_manager.py            # MPI communication wrapper
├── utils.py                   # Utilities and logging
├── validation.py              # Input validation
├── errors.py                  # Custom exceptions
│
├── core/                      # Core workflow logic
│   ├── _initialization.py
│   ├── _validation.py
│   └── _workflow.py
│
├── subproblems/               # Optimization algorithms
│   ├── base.py                # Base classes
│   ├── subproblem_manager.py  # Manager component
│   ├── subproblem_registry.py # Solver registry
│   └── registry/              # Built-in solvers
│       ├── greedy.py
│       ├── linear_knapsack.py
│       ├── quadratic_knapsack.py
│       ├── plain_single_item.py
│       └── quad_supermodular/
│           ├── quad_supermod_network.py
│           └── quad_supermod_lovasz.py
│
├── estimation/                # Estimation methods
│   ├── base.py                # Base solver class
│   ├── row_generation.py      # Row generation
│   ├── ellipsoid.py           # Ellipsoid method
│   └── inequalities.py        # Inequality solver
│
└── tests/                     # Test suite
    ├── test_greedy.py
    ├── test_knapsack.py
    ├── test_supermodular.py
    ├── test_row_generation.py
    └── ...
```

## Contributing

To extend BundleChoice:

**Add a new solver:**
1. Create file in `bundlechoice/subproblems/registry/`
2. Inherit from `SerialSubproblemBase` or `BatchSubproblemBase`
3. Implement `initialize()` and `solve()` methods
4. Register in `subproblem_registry.py`

**Add a new estimator:**
1. Create file in `bundlechoice/estimation/`
2. Inherit from `BaseEstimationSolver`
3. Implement `solve()` method
4. Add property to `BundleChoice` class

## Citation

```bibtex
@software{bundlechoice2025,
  author = {Di Pasquale, Enzo},
  title = {BundleChoice: Combinatorial Discrete Choice Estimation},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/enzodipasquale/combinatorial-choice-estimation}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Enzo Di Pasquale  
Email: [ed2189@nyu.edu](mailto:ed2189@nyu.edu)  
GitHub: [@enzodipasquale](https://github.com/enzodipasquale)
