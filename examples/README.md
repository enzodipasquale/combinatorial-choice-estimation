# BundleChoice Examples

This directory contains minimal working examples demonstrating various features of BundleChoice.

## Quick Start

All examples should be run with MPI:

```bash
mpirun -n 10 python examples/01_basic_estimation.py
```

## Examples

### 01_basic_estimation.py
**Minimal example** showing basic bundle choice estimation.
- Uses greedy algorithm
- Auto-generates features from data
- Row generation estimation
- ~70 lines

**What you'll learn:**
- Basic BundleChoice workflow
- Auto-generated feature oracles
- Data loading and scattering

### 02_custom_features.py
**Custom feature oracles** for flexible feature engineering.
- Defines custom feature function
- Supports both single and vectorized bundles
- Uses ellipsoid method
- ~90 lines

**What you'll learn:**
- Writing custom feature extraction
- Vectorized feature computation
- Multiple estimation methods

### 03_custom_subproblem.py
**Custom subproblem solver** implementation.
- Implements RandomSearchSubproblem
- Shows SerialSubproblemBase interface
- Demonstrates extensibility
- ~100 lines

**What you'll learn:**
- Creating custom optimization algorithms
- Inheriting from base classes
- Passing custom solvers to BundleChoice

### 04_mpi_usage.py
**MPI patterns and best practices**.
- Rank-specific operations
- Data distribution
- Progress monitoring
- Statistics collection
- ~120 lines

**What you'll learn:**
- Proper MPI usage
- Rank 0 vs all ranks patterns
- Validation and monitoring
- Performance profiling

### 05_advanced_config.py
**Advanced configuration** and features.
- Config profiles
- Warm start
- Result caching
- Multiple methods
- ~130 lines

**What you'll learn:**
- Using configuration profiles
- Warm start for faster solving
- Result caching for sensitivity analysis
- Comparing estimation methods

## Running Examples

### Single example:
```bash
mpirun -n 10 python examples/01_basic_estimation.py
```

### All examples:
```bash
for ex in examples/*.py; do
    echo "Running $ex..."
    mpirun -n 10 python $ex
done
```

## Requirements

- Python 3.8+
- MPI (mpi4py)
- BundleChoice installed
- Gurobi (for row generation examples)

## Next Steps

After running these examples, check out:
- `applications/` for real-world use cases
- `bundlechoice/tests/` for comprehensive tests
- Documentation for API reference

## Common Patterns

### Data Structure
```python
input_data = {
    "agent_data": {"modular": ...},  # Agent-specific features
    "item_data": {"modular": ...},   # Item-specific features
    "errors": ...,                    # Random utility shocks
    "obs_bundle": ...                 # Observed choices
}
```

### Basic Workflow
```python
bc = BundleChoice()
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()  # or set_oracle(fn)
bc.subproblems.load()
theta = bc.row_generation.solve()
```

### Validation
```python
bc.validate_setup('row_generation')  # Check before solving
```

### Monitoring
```python
def callback(info):
    print(f"Iteration {info['iteration']}: obj={info['objective']}")

theta = bc.row_generation.solve(callback=callback)
```
