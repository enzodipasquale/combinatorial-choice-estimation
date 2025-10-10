# BundleChoice User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Data Preparation](#data-preparation)
6. [Feature Engineering](#feature-engineering)
7. [Subproblem Solvers](#subproblem-solvers)
8. [Estimation Methods](#estimation-methods)
9. [MPI and Parallelization](#mpi-and-parallelization)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

BundleChoice is a toolkit for estimating discrete choice models where agents choose bundles of items. It supports:

- **Multiple estimation methods**: Row generation, ellipsoid method
- **Flexible feature engineering**: Auto-generated or custom features
- **Built-in solvers**: Greedy, knapsack, supermodular, and more
- **MPI parallelization**: Efficient distributed computation
- **Extensibility**: Easy to add custom solvers and features

### What Problems Does It Solve?

**Example Applications:**
- Export destination choice (firms choosing which countries to export to)
- Spectrum license auctions (bidders choosing license combinations)
- Product portfolio selection (firms choosing product sets)
- Multi-destination shipping (logistics choosing routes)

---

## Installation

### Basic Installation

```bash
pip install -e .
```

### Requirements

**Required:**
- Python ≥ 3.9
- NumPy ≥ 1.24
- MPI implementation (OpenMPI, MPICH)
- Gurobi ≥ 11.0 (for row generation)

**Optional:**
- Hypothesis (for property-based testing)
- pytest-mpi (for MPI testing)

### Verify Installation

```python
from bundlechoice import BundleChoice
bc = BundleChoice()
print("BundleChoice installed successfully!")
```

---

## Quick Start

### Minimal Working Example

```python
import numpy as np
from mpi4py import MPI
from bundlechoice import BundleChoice

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 1. Prepare data (on rank 0)
if rank == 0:
    num_agents, num_items, num_features = 100, 20, 5
    
    agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
    errors = np.random.normal(0, 0.1, (num_agents, num_items))
    
    input_data = {
        "agent_data": {"modular": agent_features},
        "errors": errors
    }
else:
    input_data = None

# 2. Configure BundleChoice
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

# 3. Load data
bc.data.load_and_scatter(input_data)

# 4. Set up features
bc.features.build_from_data()

# 5. Generate observations
theta_true = np.ones(5)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# 6. Add observations and estimate
if rank == 0:
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

theta_hat = bc.row_generation.solve()

if rank == 0:
    print(f"True theta: {theta_true}")
    print(f"Estimated:  {theta_hat}")
```

### Quick Setup Method

For rapid prototyping, use the `quick_setup` method:

```python
bc = BundleChoice().quick_setup(
    config=config_dict,
    input_data=data,
    features_oracle=None  # Auto-generate features
)

theta_hat = bc.row_generation.solve()
```

---

## Core Concepts

### The BundleChoice Workflow

```
1. Configure → 2. Load Data → 3. Set Features → 4. Solve Subproblems → 5. Estimate
```

**Step-by-Step:**

1. **Configure**: Set dimensions, algorithm, solver parameters
2. **Load Data**: Distribute data across MPI ranks
3. **Set Features**: Auto-generate or provide custom feature function
4. **Solve Subproblems**: Optimize bundles for given parameters
5. **Estimate**: Find parameters that match observed choices

### Key Components

- **DataManager**: Handles data distribution across MPI ranks
- **FeatureManager**: Computes features from bundles
- **SubproblemManager**: Solves optimization for each agent
- **EstimationSolver**: Estimates parameters (row generation, ellipsoid)

---

## Data Preparation

### Input Data Structure

```python
input_data = {
    "agent_data": {
        "modular": np.ndarray,      # (num_agents, num_items, num_modular_features)
        "quadratic": np.ndarray,    # (num_agents, num_items, num_items, num_quad_features) [optional]
    },
    "item_data": {
        "modular": np.ndarray,      # (num_items, num_modular_features)
        "quadratic": np.ndarray,    # (num_items, num_items, num_quad_features) [optional]
    },
    "errors": np.ndarray,           # (num_agents, num_items) or (num_simuls, num_agents, num_items)
    "obs_bundle": np.ndarray,       # (num_agents, num_items) [for estimation only]
}
```

### Data Requirements

**Required:**
- `errors`: Random utility shocks
  - 2D: `(num_agents, num_items)` for single simulation
  - 3D: `(num_simuls, num_agents, num_items)` for multiple draws

**Optional:**
- `agent_data`: Agent-specific features (at least one of modular/quadratic)
- `item_data`: Item-specific features
- `obs_bundle`: Observed choices (required for estimation, not for simulation)

### Example: Export Destination Choice

```python
# Real-world example: Firms choosing export destinations
num_firms = 1000
num_countries = 50

# Firm characteristics
firm_size = np.random.lognormal(0, 1, (num_firms, 1))  # Heterogeneous sizes

# Country characteristics  
gdp = country_data['gdp'].values.reshape(-1, 1)
distance = distance_matrix  # (num_countries, num_countries)

# Combine into input_data
input_data = {
    "agent_data": {
        "modular": np.tile(firm_size[:, None, :], (1, num_countries, 1))
    },
    "item_data": {
        "modular": gdp,
        "quadratic": distance.reshape(num_countries, num_countries, 1)
    },
    "errors": np.random.gumbel(0, 1, (num_firms, num_countries))
}
```

### Validation

Always validate your data:

```python
from bundlechoice.validation import validate_input_data_comprehensive

validate_input_data_comprehensive(input_data, dimensions_cfg)
```

**Common Issues:**
- ❌ Dimension mismatches
- ❌ NaN or Inf values
- ❌ Negative quadratic features (for supermodular problems)
- ❌ Non-zero diagonal in quadratic matrices

---

## Feature Engineering

### Auto-Generated Features

The simplest approach - BundleChoice automatically generates features:

```python
bc.features.build_from_data()
```

**How it works:**
- Detects modular/quadratic structure from data
- Generates efficient feature computation code
- Supports vectorized operations

### Custom Features

For custom utility functions, provide a feature oracle:

```python
def my_features(agent_id, bundle, data):
    """
    Compute features for a specific agent and bundle.
    
    Args:
        agent_id: Index of the agent
        bundle: Binary array (num_items,) or matrix (num_items, num_bundles)
        data: Dictionary with agent_data, item_data, errors
    
    Returns:
        Features array: (num_features,) or (num_features, num_bundles)
    """
    # Example: Combine agent preferences with bundle
    preferences = data["agent_data"]["preferences"][agent_id]
    
    if bundle.ndim == 1:
        # Single bundle
        linear = preferences @ bundle
        quadratic = -np.sum(bundle) ** 2  # Bundle size penalty
        return np.array([linear, quadratic])
    else:
        # Multiple bundles (vectorized)
        linear = preferences @ bundle
        quadratic = -np.sum(bundle, axis=0) ** 2
        return np.vstack([linear, quadratic])

bc.features.set_oracle(my_features)
```

**Important:**
- Features must be deterministic
- Should handle both single bundles and batch computation
- Return shape must be `(num_features,)` or `(num_features, num_bundles)`

### Vectorized Features

For performance, support batch computation:

```python
def vectorized_features(agent_id, bundles, data):
    """Process multiple bundles at once."""
    if bundles.ndim == 1:
        # Single bundle
        return compute_single(agent_id, bundles, data)
    else:
        # Multiple bundles - vectorized
        return compute_batch(agent_id, bundles, data)
```

---

## Subproblem Solvers

### Built-in Solvers

| Solver | Use Case | Complexity |
|--------|----------|-----------|
| `Greedy` | General purpose | O(m²) |
| `LinearKnapsack` | Linear utility + capacity | O(m log m) |
| `QuadKnapsack` | Quadratic utility + capacity | O(m²) |
| `QuadSupermodularNetwork` | Supermodular (network flow) | O(m³) |
| `QuadSupermodularLovasz` | Supermodular (Lovász ext.) | O(m²) |
| `PlainSingleItem` | Single item choice | O(m) |

**m = num_items**

### Choosing a Solver

```python
# Greedy: Fast approximation
config = {"subproblem": {"name": "Greedy"}}

# Knapsack: Exact solution with capacity constraint
config = {
    "subproblem": {
        "name": "LinearKnapsack",
        "settings": {"capacity": 10}
    }
}

# Supermodular: Exact solution for supermodular utilities
config = {"subproblem": {"name": "QuadSupermodularNetwork"}}
```

### Custom Solvers

Implement your own optimization algorithm:

```python
from bundlechoice.subproblems.base import SerialSubproblemBase

class MyCustomSolver(SerialSubproblemBase):
    def initialize(self, agent_id):
        """Initialize for specific agent."""
        # Setup any agent-specific data structures
        return None  # Or return problem state
    
    def solve(self, agent_id, theta, problem_state):
        """Solve for given parameters."""
        # Your optimization algorithm here
        bundle = your_algorithm(agent_id, theta, self.local_data)
        return bundle

# Use it
bc.subproblems.load(MyCustomSolver)
```

---

## Estimation Methods

### Row Generation (Default)

**Best for:** Large problems, exact solutions

```python
bc.load_config({
    "row_generation": {
        "max_iters": 100,
        "tolerance_optimality": 1e-6,
        "min_iters": 10,
        "gurobi_settings": {"OutputFlag": 0}
    }
})

theta_hat = bc.row_generation.solve()
```

**Parameters:**
- `max_iters`: Maximum iterations
- `tolerance_optimality`: Convergence threshold
- `min_iters`: Minimum iterations before early stopping
- `gurobi_settings`: Pass-through to Gurobi

### Ellipsoid Method

**Best for:** No Gurobi license, smaller problems

```python
bc.load_config({
    "ellipsoid": {
        "num_iters": 200,
        "initial_radius": 1.0,
        "decay_factor": 0.95
    }
})

theta_hat = bc.ellipsoid.solve()
```

**Note:** Ellipsoid is gradient-based, doesn't require LP solver.

### Monitoring Progress

Use callbacks to track estimation:

```python
def monitor(info):
    print(f"Iteration {info['iteration']}: obj={info['objective']:.4f}")

theta_hat = bc.row_generation.solve(callback=monitor)
```

---

## MPI and Parallelization

### Basic MPI Usage

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 prepares data
if rank == 0:
    input_data = prepare_data()
else:
    input_data = None

# All ranks participate in scatter
bc.data.load_and_scatter(input_data)

# All ranks solve local subproblems
local_results = bc.subproblems.solve_local(theta)

# Results gathered at rank 0
if rank == 0:
    final_result = bc.some_operation()
```

### Running with MPI

```bash
# Local machine
mpirun -n 10 python my_script.py

# HPC cluster
sbatch run_estimation.sbatch
```

### Data Distribution

BundleChoice automatically distributes agents across ranks:

```python
# Total: 1000 agents on 10 ranks
# Rank 0: agents 0-99
# Rank 1: agents 100-199
# ...
# Rank 9: agents 900-999

print(f"Rank {rank}: {bc.data.num_local_agents} local agents")
```

### Performance Tips

1. **Use buffer-based communication** (automatic for numpy arrays)
2. **Minimize broadcasts** (item_data is broadcast once)
3. **Profile communication**:
   ```python
   comm_manager = CommManager(comm, enable_profiling=True)
   profile = comm_manager.get_comm_profile()
   ```

---

## Advanced Topics

### Warm Starting

Use previous results to speed up estimation:

```python
# First estimation
theta_1 = bc.row_generation.solve()

# Use as warm start for next
bc_new = BundleChoice()
bc_new.load_config(config)
# ... setup data and features ...
theta_2 = bc_new.row_generation.solve(theta_init=theta_1)
```

### Result Caching

For sensitivity analysis:

```python
bc.subproblems.enable_cache()

# Repeated solves with same theta are cached
for theta in theta_values:
    bundles = bc.subproblems.solve_local(theta)  # Cached after first call

bc.subproblems.disable_cache()  # Clear cache
```

### Multiple Simulations

For Monte Carlo estimation:

```python
config = {
    "dimensions": {
        "num_agents": 100,
        "num_items": 20,
        "num_features": 5,
        "num_simuls": 10  # 10 simulation draws
    }
}

# Each simulation uses different error draws
# Estimation averages across simulations
```

### Configuration Presets

Quick configurations for common scenarios:

```python
# Fast prototyping
config = {
    "dimensions": {...},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 10, "tolerance_optimality": 0.01}
}

# Production
config = {
    "dimensions": {...},
    "subproblem": {"name": "QuadSupermodularNetwork"},
    "row_generation": {"max_iters": 200, "tolerance_optimality": 1e-8}
}
```

### Debugging

```python
# Check setup status
bc.print_status()

# Validate before solving
bc.validate_setup('row_generation')

# Get detailed status
status = bc.status()
print(status)
```

---

## Next Steps

- **Examples**: See `examples/` for complete workflows
- **Applications**: Check `applications/` for real-world use cases
- **Testing**: Run `pytest bundlechoice/tests/` for test suite
- **API Reference**: See class docstrings for detailed API

## Getting Help

**Common Issues:**
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Check error messages for suggestions
- Use `bc.print_status()` to diagnose setup issues

**Community:**
- GitHub Issues: Report bugs or request features
- Examples: Learn from `applications/` directory

