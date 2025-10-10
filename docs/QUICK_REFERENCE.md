# BundleChoice Quick Reference

## 🚀 Quick Start

```python
from bundlechoice import BundleChoice
import numpy as np

# 1. Setup
bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50}
})

# 2. Load data
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()

# 3. Estimate
theta_hat = bc.row_generation.solve()
```

---

## 📋 Data Format

```python
input_data = {
    "agent_data": {
        "modular": np.ndarray,    # (num_agents, num_items, num_features)
        "quadratic": np.ndarray   # (num_agents, num_items, num_items, num_features) [optional]
    },
    "item_data": {
        "modular": np.ndarray,    # (num_items, num_features)
        "quadratic": np.ndarray   # (num_items, num_items, num_features) [optional]
    },
    "errors": np.ndarray,         # (num_agents, num_items) or (num_simuls, num_agents, num_items)
    "obs_bundle": np.ndarray      # (num_agents, num_items) [for estimation]
}
```

---

## 🎛️ Configuration

### Minimal Config
```python
config = {
    "dimensions": {
        "num_agents": 100,
        "num_items": 20,
        "num_features": 5
    },
    "subproblem": {"name": "Greedy"}
}
```

### Complete Config
```python
config = {
    "dimensions": {
        "num_agents": 100,
        "num_items": 20,
        "num_features": 5,
        "num_simuls": 1
    },
    "subproblem": {
        "name": "Greedy",
        "settings": {}
    },
    "row_generation": {
        "max_iters": 100,
        "tolerance_optimality": 1e-6,
        "min_iters": 10,
        "gurobi_settings": {"OutputFlag": 0}
    },
    "ellipsoid": {
        "num_iters": 200,
        "initial_radius": 1.0,
        "decay_factor": 0.95
    }
}
```

---

## 🔧 Subproblem Solvers

| Solver | Command | Use Case |
|--------|---------|----------|
| Greedy | `{"name": "Greedy"}` | Fast, general purpose |
| Linear Knapsack | `{"name": "LinearKnapsack", "settings": {"capacity": 10}}` | Linear + capacity |
| Quad Knapsack | `{"name": "QuadKnapsack", "settings": {"capacity": 10}}` | Quadratic + capacity |
| Supermodular (Network) | `{"name": "QuadSupermodularNetwork"}` | Supermodular, larger problems |
| Supermodular (Lovász) | `{"name": "QuadSupermodularLovasz"}` | Supermodular, smaller problems |
| Single Item | `{"name": "PlainSingleItem"}` | Single item choice |

---

## 🎨 Features

### Auto-Generated
```python
bc.features.build_from_data()
```

### Custom Features
```python
def my_features(agent_id, bundle, data):
    # Single bundle
    if bundle.ndim == 1:
        return np.array([feature1, feature2, ...])
    # Batch of bundles
    else:
        return np.vstack([feature1, feature2, ...])

bc.features.set_oracle(my_features)
```

---

## 📊 Estimation Methods

### Row Generation
```python
theta_hat = bc.row_generation.solve()

# With callback
def monitor(info):
    print(f"Iter {info['iteration']}: {info['objective']}")

theta_hat = bc.row_generation.solve(callback=monitor)
```

### Ellipsoid Method
```python
theta_hat = bc.ellipsoid.solve()
```

---

## 🔍 Validation & Debugging

### Check Setup
```python
bc.print_status()
```

### Validate Setup
```python
bc.validate_setup('row_generation')
```

### Get Status Dict
```python
status = bc.status()
print(status['config_loaded'])
print(status['data_loaded'])
```

### Validate Data
```python
from bundlechoice.validation import validate_input_data_comprehensive

validate_input_data_comprehensive(input_data, dimensions_cfg)
```

---

## 🌐 MPI Usage

### Basic Pattern
```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Rank 0 prepares data
if rank == 0:
    input_data = prepare_data()
else:
    input_data = None

# All ranks participate
bc.data.load_and_scatter(input_data)
```

### Run with MPI
```bash
mpirun -n 10 python script.py
```

---

## ⚠️ Common Errors

### SetupError
```python
# Problem: Missing setup steps
# Solution: Follow correct order
bc.load_config(config)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
```

### DimensionMismatchError
```python
# Problem: Data doesn't match config
# Solution: Check dimensions
print(agent_data.shape)  # Should be (num_agents, num_items, ...)
print(errors.shape)      # Should be (num_agents, num_items)
```

### DataError (NaN/Inf)
```python
# Problem: Invalid values in data
# Solution: Clean data
data = np.nan_to_num(data, nan=0.0)
data = np.clip(data, -1e10, 1e10)
```

---

## 🎯 Complete Workflow

### 1. Data Preparation
```python
# Prepare data
agent_data = ...
item_data = ...
errors = np.random.normal(0, sigma, (num_agents, num_items))

input_data = {
    "agent_data": {"modular": agent_data},
    "item_data": {"modular": item_data},
    "errors": errors
}
```

### 2. Configuration
```python
config = {
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 50}
}
```

### 3. Setup & Validation
```python
bc = BundleChoice()
bc.load_config(config)

# Validate data
from bundlechoice.validation import validate_input_data_comprehensive
validate_input_data_comprehensive(input_data, bc.config.dimensions)

bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
```

### 4. Generate Observations
```python
theta_true = np.array([1.0, 0.5, 1.2, ...])
obs_bundles = bc.subproblems.init_and_solve(theta_true)

if rank == 0:
    input_data["obs_bundle"] = obs_bundles
```

### 5. Reload & Estimate
```python
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()

theta_hat = bc.row_generation.solve()
```

### 6. Validate Results
```python
if rank == 0:
    error = np.linalg.norm(theta_hat - theta_true)
    print(f"Estimation error: {error:.4f}")
    
    bundles_check = bc.subproblems.init_and_solve(theta_hat)
    match_rate = (bundles_check == obs_bundles).mean()
    print(f"Match rate: {match_rate*100:.1f}%")
```

---

## 💾 Save & Load

### Save Results
```python
if rank == 0:
    np.save('theta_hat.npy', theta_hat)
    np.savez('results.npz', 
             theta=theta_hat, 
             bundles=obs_bundles,
             config=config)
```

### Save Data
```python
np.savez_compressed('input_data.npz',
    agent_modular=input_data['agent_data']['modular'],
    errors=input_data['errors'],
    obs_bundle=input_data['obs_bundle']
)
```

### Load Data
```python
data = np.load('input_data.npz')
input_data = {
    "agent_data": {"modular": data['agent_modular']},
    "errors": data['errors'],
    "obs_bundle": data['obs_bundle']
}
```

---

## 🔧 Advanced Features

### Quick Setup
```python
bc = BundleChoice().quick_setup(config, input_data, features_oracle=None)
theta_hat = bc.row_generation.solve()
```

### Warm Start
```python
theta_hat = bc.row_generation.solve(theta_init=theta_previous)
```

### Enable Caching
```python
bc.subproblems.enable_cache()
# Repeated solves with same theta are cached
```

### Temporary Config
```python
with bc.temp_config(row_generation={'max_iters': 10}):
    quick_theta = bc.row_generation.solve()
# Config restored after
```

### Generate & Reload
```python
obs_bundles = bc.generate_observations(theta_true)
# Automatically reloads data with observations
```

---

## 📚 Documentation Links

- **[User Guide](USER_GUIDE.md)** - Complete guide
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common problems
- **[Best Practices](BEST_PRACTICES.md)** - Production patterns
- **[Examples](../examples/)** - Working code

---

## 🆘 Getting Help

### 1. Check error message
```
SetupError: ...
💡 Suggestion: [actionable fix]
```

### 2. Validate setup
```python
bc.validate_setup('row_generation')
```

### 3. Print status
```python
bc.print_status()
```

### 4. Check docs
- [Troubleshooting](TROUBLESHOOTING.md)
- [User Guide](USER_GUIDE.md)

### 5. GitHub Issues
https://github.com/enzodipasquale/combinatorial-choice-estimation/issues

