# BundleChoice Documentation

Welcome to the BundleChoice documentation! This guide will help you understand, use, and extend BundleChoice for your combinatorial choice estimation needs.

## 📚 Documentation Structure

### Getting Started
- **[User Guide](USER_GUIDE.md)** - Comprehensive guide from installation to advanced features
- **[Quick Start](../README.md#usage)** - Minimal working example (in main README)
- **[Examples](../examples/README.md)** - Working code examples

### Troubleshooting & Best Practices
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common errors and solutions
- **[Best Practices](BEST_PRACTICES.md)** - Code organization, performance, and deployment

### Reference
- **[API Documentation](#)** - Auto-generated API reference (coming soon)
- **[Configuration Reference](../examples/05_advanced_config.py)** - All configuration options

## 🚀 Quick Navigation

### I'm New to BundleChoice
1. Read the [User Guide](USER_GUIDE.md) introduction
2. Follow the [Quick Start](../README.md#usage) example
3. Run examples in `examples/` directory
4. Check [Troubleshooting](TROUBLESHOOTING.md) if you hit issues

### I Have a Specific Problem
- **Setup issues**: [Troubleshooting → Setup Issues](TROUBLESHOOTING.md#setup-issues)
- **Data problems**: [Troubleshooting → Data Problems](TROUBLESHOOTING.md#data-problems)
- **Performance**: [Best Practices → Performance](BEST_PRACTICES.md#performance-optimization)
- **MPI errors**: [Troubleshooting → MPI Problems](TROUBLESHOOTING.md#mpi-problems)

### I Want to Extend BundleChoice
- **Custom features**: [User Guide → Feature Engineering](USER_GUIDE.md#feature-engineering)
- **Custom solvers**: [User Guide → Subproblem Solvers](USER_GUIDE.md#subproblem-solvers)
- **Code organization**: [Best Practices → Code Organization](BEST_PRACTICES.md#code-organization)

### I'm Deploying to Production
- **Cluster deployment**: [Best Practices → Production Deployment](BEST_PRACTICES.md#production-deployment)
- **Performance tuning**: [Best Practices → Performance Optimization](BEST_PRACTICES.md#performance-optimization)
- **Error handling**: [Best Practices → Error Handling](BEST_PRACTICES.md#error-handling)

## 📖 Core Concepts

### The BundleChoice Workflow
```
Configure → Load Data → Set Features → Solve Subproblems → Estimate
```

**Key Components:**
- **DataManager**: Distributes data across MPI ranks
- **FeatureManager**: Computes features from bundles
- **SubproblemManager**: Solves optimization for each agent
- **EstimationSolver**: Estimates parameters via row generation or ellipsoid method

### When to Use Which Solver?

| Your Problem | Recommended Solver | Why |
|--------------|-------------------|-----|
| General purpose, fast | `Greedy` | O(m²) approximation |
| Linear + capacity | `LinearKnapsack` | Exact, O(m log m) |
| Quadratic + capacity | `QuadKnapsack` | Exact, O(m²) |
| Supermodular | `QuadSupermodularNetwork` | Exact, network flow |
| Small supermodular | `QuadSupermodularLovasz` | Exact, Lovász extension |
| Single item | `PlainSingleItem` | Trivial, O(m) |

### Estimation Methods

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **Row Generation** | Exact, scalable | Needs Gurobi | Production, accuracy |
| **Ellipsoid** | No license needed | Slower convergence | Development, no Gurobi |

## 🔧 Common Workflows

### Basic Estimation
```python
from bundlechoice import BundleChoice

bc = BundleChoice().quick_setup(config, input_data)
theta_hat = bc.row_generation.solve()
```

### Custom Features
```python
def my_features(agent_id, bundle, data):
    return custom_computation(agent_id, bundle, data)

bc.features.set_oracle(my_features)
```

### MPI Deployment
```bash
mpirun -n 20 python my_script.py
```

## 🆘 Getting Help

### Error Messages
BundleChoice provides helpful error messages with suggestions:
```
SetupError: Cannot initialize subproblem manager
💡 Suggestion:
  Complete these steps in order:
  1. bc.load_config(config_dict)
  2. bc.data.load_and_scatter(input_data)
  3. bc.features.build_from_data()
```

### Diagnostic Tools
```python
# Check setup status
bc.print_status()

# Validate before solving
bc.validate_setup('row_generation')

# Get detailed status
status = bc.status()
```

### Resources
- **GitHub Issues**: [Report bugs or request features](https://github.com/enzodipasquale/combinatorial-choice-estimation/issues)
- **Examples**: See `examples/` directory for working code
- **Applications**: Check `applications/` for real-world use cases

## 📝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Requesting features
- Contributing code
- Writing documentation

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

---

## Quick Links

### Documentation
- [User Guide](USER_GUIDE.md) - Complete usage guide
- [Troubleshooting](TROUBLESHOOTING.md) - Common problems and solutions
- [Best Practices](BEST_PRACTICES.md) - Production-ready code patterns

### Code
- [Examples](../examples/) - Working examples
- [Applications](../applications/) - Real-world use cases
- [Tests](../bundlechoice/tests/) - Test suite

### Research
- Main README: [README.md](../README.md)
- Applications README: [applications/README.md](../applications/)

