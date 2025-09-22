# Performance Comparisons

This folder contains experiments for comparing the performance of different formulations and algorithms in the BundleChoice framework.

## Files

### `row_generation_1slack_vs_standard.py`
Comprehensive performance comparison between the 1slack and standard row generation formulations.

**Features:**
- Tests multiple subproblem types (Greedy, LinearKnapsack, PlainSingleItem)
- Tests different problem sizes (varying agents, items, features)
- Measures solve time, parameter recovery accuracy, convergence
- Generates detailed JSON reports with timestamps
- Calculates speedup ratios and performance metrics

**Usage:**
```bash
mpirun -n 10 python experiments/performance_comparisons/row_generation_1slack_vs_standard.py
```

### `quick_comparison.py`
Simplified version for quick testing and development.

**Features:**
- Single scenario (Greedy subproblem, 1000 agents, 100 items)
- Fast execution for development and debugging
- Console output with key metrics

**Usage:**
```bash
mpirun -n 10 python experiments/performance_comparisons/quick_comparison.py
```

## Metrics Compared

- **Solve Time**: Total wall-clock time for parameter estimation
- **Parameter Recovery**: L2 norm error and max relative error vs true parameters
- **Convergence**: Whether the algorithm converged within tolerance
- **Memory Usage**: Number of constraints in master problem (for standard formulation)
- **Objective Value**: Final objective value of the master problem

## Expected Results

The 1slack formulation should generally be:
- **Faster**: Fewer variables and constraints in master problem
- **More Memory Efficient**: Single utility variable vs one per simulation/agent
- **Similar Accuracy**: Should recover parameters with similar accuracy

## Output

Results are saved as JSON files with timestamps in the same directory, containing:
- Detailed timing and accuracy metrics for each formulation
- Problem parameters and configuration
- Summary statistics and speedup ratios

