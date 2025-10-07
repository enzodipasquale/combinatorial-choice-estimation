# Combinatorial Discrete Choice Model Estimation

Parametric estimation for combinatorial discrete choice models using row generation and ellipsoid methods. Supports MPI parallelization and includes 8 optimization algorithms for solving bundle choice subproblems.

---

## Overview

The estimation procedure needs two oracles:

**Features Oracle:** Computes feature vectors for bundles. Either write your own function or use `build_from_data()` to auto-generate from modular/quadratic data structures.

**Demand Oracle:** Solves the bundle optimization subproblem. Use one of 8 built-in algorithms or write a custom solver by inheriting from `BaseSubproblem`.

**Built-in algorithms:**
- `Greedy`, `OptimizedGreedy`, `GreedyJIT` - greedy heuristics with vectorization
- `QuadSupermodularNetwork`, `QuadSupermodularLovasz` - min-cut and Lovász extension for supermodular functions
- `LinearKnapsack`, `QuadKnapsack` - knapsack variants
- `PlainSingleItem` - single-item choice

---

## Implementation Details

**Estimation methods:**
- Row generation (uses Gurobi for master problem)
- Ellipsoid method

**Key features:**
- MPI parallelization with `mpi4py`
- Distributed data handling across ranks
- Extensible subproblem registry
- Full test suite

---

## Current Status

**Working:**
- Row generation and ellipsoid estimation methods
- 8 subproblem algorithms with MPI support
- Auto-generated and custom feature oracles
- Full test coverage

**In progress:**
- Performance benchmarking
- Documentation improvements

---

## Installation

To set up the environment and install all dependencies, run the provided setup script:

```bash
./dev_setup.sh
```

This will:
- Create and activate a virtual environment (`.bundle/`)
- Install all required dependencies
- Set up MPI environment for parallel computing

---

## Quick Start

```python
from bundlechoice import BundleChoice
import numpy as np

# Create and configure
bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 50, "num_features": 10, "num_simuls": 1},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50}
})

# Load data (on rank 0)
bc.data.load_and_scatter(input_data)

# Set up features - either auto-generate:
bc.features.build_from_data()

# Or define custom:
def my_features(agent_id, bundle, data):
    return data["agent_data"]["features"][agent_id] @ bundle
bc.features.set_oracle(my_features)

# Estimate parameters
theta_hat = bc.row_generation.solve()
```

Run with MPI: `mpirun -n 10 python your_script.py`



## Contributing

Contributions welcome. When contributing:

- Include tests for new code
- Maintain MPI compatibility for core functionality
- Follow existing patterns
- Use descriptive names (no abbreviations)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Developed by Enzo Di Pasquale  
GitHub: [enzodipasquale](https://github.com/enzodipasquale)
