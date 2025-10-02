# Performance Comparisons

This folder contains clean experiments for comparing the performance of standard and 1slack row generation formulations.

## Files

### `test_greedy_comparison.py`
Compares standard vs 1slack formulations for the Greedy subproblem.
- Based on `test_estimation_row_generation_1slack_greedy.py`
- Tests with 250 agents, 50 items, 6 features
- Compares objective values, solve times, and parameter recovery

### `test_linear_knapsack_comparison.py`
Compares standard vs 1slack formulations for the LinearKnapsack subproblem.
- Based on `test_estimation_row_generation_1slack_linear_knapsack.py`
- Tests with 250 agents, 20 items, 4 features
- Compares objective values, solve times, and parameter recovery

### `test_plain_single_item_comparison.py`
Compares standard vs 1slack formulations for the PlainSingleItem subproblem.
- Based on `test_estimation_row_generation_1slack_plain_single_item.py`
- Tests with 500 agents, 2 items, 5 features
- Compares objective values, solve times, and parameter recovery

### `test_quad_supermodular_comparison.py`
Compares standard vs 1slack formulations for the QuadSupermodularNetwork subproblem.
- Based on `test_estimation_row_generation_1slack_quadratic_supermodular.py`
- Tests with 250 agents, 50 items, 6 features
- Compares objective values, solve times, and parameter recovery

## Usage

Run individual experiments:
```bash
mpirun -n 10 python experiments/performance_comparisons/test_greedy_comparison.py
mpirun -n 10 python experiments/performance_comparisons/test_linear_knapsack_comparison.py
mpirun -n 10 python experiments/performance_comparisons/test_plain_single_item_comparison.py
mpirun -n 10 python experiments/performance_comparisons/test_quad_supermodular_comparison.py
```

## Metrics Compared

- **Objective Value**: Final objective value of the master problem
- **Solve Time**: Total wall-clock time for parameter estimation
- **Parameter Recovery**: L2 norm error and max relative error vs true parameters
- **Iterations**: Number of row generation iterations
- **Constraints**: Number of constraints in final master problem

## Expected Results

The two formulations should produce **identical objective values** (mathematical equivalence).
Any differences indicate implementation issues that need to be resolved.
