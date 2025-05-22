# Combinatorial Discrete Choice Model Estimation

🚧 **Work In Progress** 🚧

This repository implements a parametric estimation method for combinatorial discrete choice models. It uses user-provided oracles and advanced optimization techniques along with parallel computing to efficiently estimate model parameters.

---

## Overview

The estimation procedure requires two user-provided oracles (functions or class methods):

- **Features Oracle:** A user-defined function or class method that computes the feature vector for each bundle.
- **Demand Oracle:** A user-defined function or class method that computes the demanded bundle at candidate parameter values.

The demand oracle supports integration with **Mixed Integer Programming (MIP) solvers** for solving complex subproblems. The repository also includes a library of built-in subproblem solvers such as:

- Greedy algorithms
- Quadratic supermodular knapsack
- Linear knapsack
- Unconstrained supermodular optimization

---

## Implementation Details

- The core estimation algorithm is based on **row-generation**, which is fully implemented and functional.
- An **ellipsoid method** for estimation is under active development.
- The code supports **parallel computing using `mpi4py`**, designed for efficient execution on High-Performance Computing (HPC) systems.
- Distributed data handling across MPI ranks, integration with Gurobi for MIP solving, and utilities for logging and problem updates are included.

---

## Current Status

- Fully functional row-generation estimation method with parallel subproblem solving.
- Integration of user-defined demand and features oracles.
- Built-in library of subproblem solvers.
- Ellipsoid method under development.
- Documentation and usage examples forthcoming.

---

## Installation

To set up the environment and install all dependencies, run the provided setup script:

```bash
./dev_setup.sh

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Developed by Enzo Di Pasquale  
GitHub: [enzodipasquale](https://github.com/enzodipasquale)
