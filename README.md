# Combinatorial Choice Estimation

Toolkit for estimating discrete choice models where agents choose bundles of items (export destinations, product portfolios, spectrum licenses, etc.). Uses row generation or ellipsoid methods with MPI parallelization.

## What it does

You have agents making combinatorial choices. You observe their choices and want to estimate preference parameters. This estimates them.

**Two things you provide:**
1. **How to compute features** from bundles (or use auto-generated features)
2. **How to solve the optimization** (use built-in algorithms or write your own)

**Built-in solvers:**
- Greedy (3 variants: standard, optimized, JIT)  
- Knapsack (linear, quadratic)
- Supermodular (network flow, Lovász extension)
- Single-item choice

## Installation

```bash
pip install -e .
```

Requires Gurobi (for row generation) and MPI (for parallelization).

## Usage

```python
from bundlechoice import BundleChoice

bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50}
})

# Load data
bc.data.load_and_scatter(input_data)

# Either auto-generate features:
bc.features.build_from_data()

# Or provide custom feature function:
def my_features(agent_id, bundle, data):
    return data["agent_data"]["X"][agent_id] @ bundle

bc.features.set_oracle(my_features)

# Estimate
theta = bc.row_generation.solve()
```

Run with MPI: `mpirun -n 10 python script.py`

## Examples

See `examples/` for:
- Basic estimation (70 lines)
- Custom features (90 lines)  
- Custom optimization algorithm (100 lines)
- MPI usage patterns

## Applications

Three real applications in `applications/`:

**Gravity model (`gravity/`):** Export destination choice with real country data from World Bank API. Features include GDP, distance, trade costs, language similarity.

**Spectrum auctions (`combinatorial_auction/`):** FCC spectrum license allocation using BTA data.

**Firm exports (`firms_export/`):** Multi-destination export decisions from Stata firm data.

## Customization

### Custom features
```python
def features(agent_id, bundle, data):
    # Your feature engineering
    return feature_vector

bc.features.set_oracle(features)
```

### Custom optimization
```python
from bundlechoice.subproblems.base import SerialSubproblemBase

class MyOptimizer(SerialSubproblemBase):
    def initialize(self, agent_id):
        # Setup
        return problem_state
    
    def solve(self, agent_id, theta, problem_state):
        # Solve and return bundle
        return optimal_bundle

bc.subproblems.load(MyOptimizer)
```

## How it works

**Row generation:** Master problem (Gurobi LP) + separation oracle (your optimization algorithm). Iteratively adds violated constraints until convergence.

**Ellipsoid method:** Gradient-based parameter search. Slower but doesn't need Gurobi.

**MPI parallelization:** Each rank handles subset of agents. Features and subproblems solved in parallel, results gathered at rank 0.

## Debugging

Check setup:
```python
bc.print_status()
# Shows what's initialized, what's missing

# Validate before solving
bc.validate_setup('row_generation')

# Get detailed status
status = bc.status()
```

## Documentation

**📚 Comprehensive guides available in `docs/`:**

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage guide from basics to advanced
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Cheat sheet for common tasks
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common errors and solutions
- **[Best Practices](docs/BEST_PRACTICES.md)** - Production-ready patterns
- **[Documentation Index](docs/README.md)** - Navigate all documentation

**New Features:**
- ✨ Enhanced error messages with actionable suggestions
- ✨ Comprehensive validation framework
- ✨ Property-based testing suite
- ✨ 100+ new tests for better reliability

## Tests

```bash
# Run all tests
pytest bundlechoice/tests/

# Run property-based tests
pytest bundlechoice/tests/test_property_based.py -v

# Run integration tests with MPI
mpirun -n 4 pytest bundlechoice/tests/test_integration_comprehensive.py -v
```

**100+ tests** covering:
- All solvers and estimation methods
- Property-based tests (infinite edge cases via Hypothesis)
- Complete workflow integration tests
- Edge case and boundary condition tests
- Error handling and validation

## License

MIT

## Contact

Enzo Di Pasquale  
[enzodipasquale](https://github.com/enzodipasquale)
