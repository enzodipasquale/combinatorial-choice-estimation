# BundleChoice

A Python toolkit for estimating discrete choice models when agents select bundles of items instead of single choices. Think firms choosing which countries to export to, bidders selecting spectrum licenses, or consumers picking product portfolios.

## Why this exists

Standard discrete choice models (like logit) work when people choose one thing from a menu. But many real decisions involve choosing multiple things simultaneously—firms don't just export to one country, they export to several. The question is: given observed choices, can we estimate the underlying preferences?

This toolkit handles that estimation problem. You provide the observed choices and specify how to compute utility, and it estimates the preference parameters using either row generation (fast, needs Gurobi) or ellipsoid methods (slower, no commercial solver needed).

## What you need to provide

Two things:

1. **Feature function**: How to compute utility features from a bundle
2. **Optimization algorithm**: How agents solve their utility maximization problem

For the second part, you can use built-in solvers or write your own.

### Built-in solvers

- **Greedy** (3 variants): For problems where agents greedily add items
- **Knapsack**: Linear or quadratic utility with capacity constraints  
- **Supermodular**: Network flow or Lovász extension for complementarities
- **Single-item**: Standard discrete choice as a special case

## Installation

```bash
pip install -e .
```

You'll need MPI installed on your system for parallelization, and Gurobi (with license) if you want to use row generation.

## Basic example

```python
from bundlechoice import BundleChoice
import numpy as np

# Generate some data
num_agents = 100
num_items = 20
num_features = 5

agent_features = np.random.normal(0, 1, (num_agents, num_items, num_features))
errors = np.random.normal(0, 0.1, (1, num_agents, num_items))

input_data = {
    "agent_data": {"modular": agent_features},
    "errors": errors
}

# Set up the model
bc = BundleChoice()
bc.load_config({
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": 1
    },
    "subproblem": {"name": "Greedy"},
    "row_generation": {
        "max_iters": 50,
        "tolerance_optimality": 0.001
    }
})

# Load data and estimate
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# Generate observed choices (in practice, you'd have these from data)
theta_true = np.ones(num_features)
obs_bundles = bc.subproblems.init_and_solve(theta_true)

# Add observations and re-estimate
if bc.is_root():
    input_data["obs_bundle"] = obs_bundles

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

# Estimate parameters
theta_hat = bc.row_generation.solve()
```

Run with MPI:
```bash
mpirun -n 10 python script.py
```

## Custom features

The auto-generated features work for simple cases, but you'll often want custom feature engineering:

```python
def my_features(agent_id, bundle, data):
    """
    Compute features for a given agent and bundle.
    
    Args:
        agent_id: Index of the agent
        bundle: Binary array indicating which items are chosen
        data: Local data dictionary from DataManager
    
    Returns:
        Feature vector (numpy array)
    """
    # Extract agent-specific data
    X = data["agent_data"]["modular"][agent_id]  # shape: (num_items, num_features)
    
    # Compute features (e.g., sum of item characteristics)
    features = X.T @ bundle  # shape: (num_features,)
    
    return features

bc.features.set_oracle(my_features)
```

## Custom optimization

If the built-in solvers don't fit your problem, write your own:

```python
from bundlechoice.subproblems.base import SerialSubproblemBase
import numpy as np

class MyCustomSolver(SerialSubproblemBase):
    def initialize(self, agent_id):
        """Set up the problem for a specific agent."""
        # Return any problem-specific state you need
        data = self.local_data
        capacity = data["agent_data"]["capacity"][agent_id]
        weights = data["item_data"]["weights"]
        return {"capacity": capacity, "weights": weights}
    
    def solve(self, agent_id, theta, problem_state):
        """Solve the optimization problem given parameters theta."""
        # Your custom optimization logic here
        capacity = problem_state["capacity"]
        weights = problem_state["weights"]
        
        # Compute utilities
        utilities = np.zeros(self.num_items)
        for j in range(self.num_items):
            bundle = np.zeros(self.num_items)
            bundle[j] = 1
            features = self.features_oracle(agent_id, bundle)
            utilities[j] = theta @ features
        
        # Select items greedily until capacity is reached
        selected = np.zeros(self.num_items, dtype=bool)
        current_weight = 0
        
        for j in np.argsort(-utilities):
            if current_weight + weights[j] <= capacity:
                selected[j] = True
                current_weight += weights[j]
        
        return selected.astype(float)

# Use it
bc.subproblems.load(MyCustomSolver)
```

## Real applications

The `applications/` directory has three real-world examples:

### Gravity model (export destinations)

Firms choosing which countries to export to based on GDP, distance, and trade costs. Uses World Bank data and the quadratic supermodular solver.

```bash
cd applications/gravity
python 1_generate_data.py    # Fetch country data from World Bank API
python 2_simulate.py          # Simulate firm export choices
python 3_visualize.py         # Create plots and summary stats
```

**Key result**: 94% of firms export to India and China (high GDP countries), average firm exports to 20 countries, strong correlation with gravity model predictions.

### Spectrum auctions

FCC spectrum license allocation using Business Trading Area (BTA) data. Bidders select portfolios of geographically adjacent licenses.

### Firm exports (Mexico)

Multi-destination export decisions from real firm-level data. Mexican firms overwhelmingly choose USA as primary destination (100% in simulation), which matches NAFTA/USMCA patterns.

## How it works under the hood

### Row generation

The standard approach for revealed preference estimation. The algorithm iteratively solves:

1. **Master problem** (LP): Finds parameters satisfying rationality constraints
2. **Separation oracle**: Checks if current parameters satisfy optimality for all agents
3. **Add violated constraints**: If any agent could do better, add that constraint

This continues until no violations exist (converged) or max iterations reached. Uses Gurobi for the master problem, your subproblem solver for the oracle.

### Ellipsoid method

Alternative that doesn't need Gurobi. Uses subgradient information to shrink an ellipsoid around the feasible parameter region. Slower but more flexible.

### MPI parallelization  

Data gets scattered across ranks at initialization. Each rank handles a subset of agents. When solving:

- Each rank computes features and solves subproblems for its agents
- Results gather back to rank 0 for the master problem
- Parameters broadcast back to all ranks

This scales well—the benchmarks directory shows good speedup up to 20+ processes.

## Configuration

Settings go in a YAML file or dictionary:

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
```

Load it:
```python
bc.load_config("config.yaml")
# or
bc.load_config(config_dict)
```

## Workflow shortcuts

For common patterns:

```python
# Quick setup
bc.quick_setup(config, input_data, features_oracle=None)
theta = bc.row_generation.solve()

# Generate observations from true parameters
bc.generate_observations(theta_true)

# Temporary config changes
with bc.temp_config(row_generation={'max_iters': 5}):
    quick_theta = bc.row_generation.solve()

# Check status
bc.print_status()
```

## Examples

The `examples/` directory has 5 progressively complex examples:

1. **Basic estimation** (70 lines): Minimal working example
2. **Custom features** (90 lines): Write your own feature function
3. **Custom subproblem** (100 lines): Implement a custom solver
4. **MPI usage** (120 lines): Patterns for distributed computing
5. **Advanced config** (130 lines): Warm starts, caching, multiple methods

Run any example:
```bash
mpirun -n 10 python examples/01_basic_estimation.py
```

## Testing

25+ tests covering all solvers and estimation methods:

```bash
# Run all tests (excluding ellipsoid, which is slow)
mpirun -n 10 pytest bundlechoice/tests/ -k "not ellipsoid"

# Run specific solver tests
mpirun -n 10 pytest bundlechoice/tests/test_greedy.py

# Run with coverage
mpirun -n 10 pytest bundlechoice/tests/ --cov=bundlechoice
```

## Debugging

If something's not working:

```python
# Check what's initialized
bc.print_status()

# Validate before solving
bc.validate_setup('row_generation')

# Get detailed status
status = bc.status()
print(status['data_loaded'])  # True/False
print(status['features_set'])  # True/False
```

Common issues:

- **"Cannot initialize subproblem manager"**: You haven't loaded data or set features yet
- **Gurobi license error**: Need valid Gurobi license for row generation (use ellipsoid method as alternative)
- **MPI errors**: Make sure you're running with `mpirun -n <num_processes>`

## Performance

From the benchmarking results:

- **Greedy**: ~0.5s per iteration (100 agents, 50 items)
- **Knapsack**: ~2s per iteration  
- **Supermodular**: ~5s per iteration
- **Convergence**: Usually 20-50 iterations for synthetic data

MPI scaling is nearly linear up to ~20 processes, then communication overhead starts to matter.

## Project structure

```
bundlechoice/
├── core.py              # Main BundleChoice class
├── data_manager.py      # MPI data distribution
├── feature_manager.py   # Feature extraction
├── config.py           # Configuration dataclasses
├── subproblems/
│   ├── base.py         # Base classes for solvers
│   ├── registry/       # Built-in solvers
│   └── subproblem_manager.py
├── estimation/
│   ├── row_generation.py
│   ├── ellipsoid.py
│   └── inequalities.py
└── tests/              # Test suite

applications/
├── gravity/            # Export destination choice
├── combinatorial_auction/  # Spectrum licenses
└── firms_export/       # Firm-level exports

examples/               # Learning examples
benchmarking/          # Performance tests
```

## Contributing

The code is modular by design. To add:

- **New solver**: Subclass `SerialSubproblemBase` or `BatchSubproblemBase` in `bundlechoice/subproblems/registry/`
- **New estimator**: Subclass `BaseEstimationSolver` in `bundlechoice/estimation/`  
- **New application**: Create a directory in `applications/` with config.yaml and run scripts

## Requirements

- Python ≥3.9
- NumPy ≥1.24
- SciPy ≥1.10
- mpi4py ≥3.1
- gurobipy ≥11.0 (optional, for row generation)
- matplotlib ≥3.7 (for visualization)
- networkx ≥3.0 (for supermodular solvers)
- PyYAML ≥6.0

## License

MIT

## Citation

If you use this in research:

```bibtex
@software{bundlechoice2025,
  author = {Di Pasquale, Enzo},
  title = {BundleChoice: Combinatorial Discrete Choice Estimation},
  year = {2025},
  url = {https://github.com/enzodipasquale/combinatorial-choice-estimation}
}
```

## Contact

Enzo Di Pasquale  
[ed2189@nyu.edu](mailto:ed2189@nyu.edu)  
[github.com/enzodipasquale](https://github.com/enzodipasquale)
