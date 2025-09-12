# Combinatorial Discrete Choice Model Estimation

A comprehensive implementation of parametric estimation methods for combinatorial discrete choice models. This repository provides efficient estimation algorithms using user-provided oracles and advanced optimization techniques, with full support for parallel computing using MPI.

---

## Overview

The estimation procedure requires two user-provided oracles (functions or class methods):

- **Features Oracle:** A user-defined function or class method that computes the feature vector for each bundle.
- **Demand Oracle:** A user-defined function or class method that computes the demanded bundle at candidate parameter values.

The demand oracle supports integration with **Mixed Integer Programming (MIP) solvers** for solving complex subproblems. The repository includes a comprehensive library of built-in subproblem solvers:

- **Greedy algorithms** - Fast approximate solutions for various problem types
- **Quadratic supermodular optimization** - Network flow and Lovász extension methods
- **Linear knapsack** - Standard knapsack problem solver
- **Plain single item** - Basic single-item choice optimization
- **Unconstrained supermodular optimization** - Specialized algorithms for supermodular functions

---

## Implementation Details

### Estimation Methods

- **Row Generation Method** - Fully implemented and tested with parallel subproblem solving
- **Ellipsoid Method** - Complete implementation with comprehensive test coverage

### Core Features

- **Parallel Computing** - Full MPI support using `mpi4py` for High-Performance Computing (HPC) systems
- **Distributed Data Handling** - Efficient data distribution across MPI ranks
- **Gurobi Integration** - Seamless integration with Gurobi for MIP solving
- **Modular Architecture** - Clean separation of concerns with extensible subproblem registry
- **Comprehensive Testing** - Full test suite covering all estimation methods and subproblems

---

## Current Status

✅ **Fully Implemented and Tested:**
- Row-generation estimation method with parallel subproblem solving
- Ellipsoid method for estimation with full test coverage
- Complete library of subproblem solvers
- MPI-based parallel computing infrastructure
- User-defined demand and features oracles integration
- Comprehensive testing framework

🚧 **Under Development:**
- Additional optimization algorithms
- Enhanced documentation and usage examples
- Performance benchmarking and optimization

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



## Contributing

Contributions are welcome! Please open issues or submit pull requests. The project follows these guidelines:

- All new code should include comprehensive tests
- MPI compatibility is required for core functionality
- Follow the existing code structure and patterns
- Use descriptive attribute names over abbreviations

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Developed by Enzo Di Pasquale  
GitHub: [enzodipasquale](https://github.com/enzodipasquale)
