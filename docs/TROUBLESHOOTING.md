# BundleChoice Troubleshooting Guide

## Table of Contents
1. [Common Errors](#common-errors)
2. [Setup Issues](#setup-issues)
3. [Data Problems](#data-problems)
4. [Performance Issues](#performance-issues)
5. [MPI Problems](#mpi-problems)
6. [Solver Failures](#solver-failures)
7. [Best Practices](#best-practices)

---

## Common Errors

### SetupError: Missing Required Setup

**Error Message:**
```
SetupError: Cannot initialize subproblem manager - missing required setup
Missing:
  ✗ data (call bc.data.load_and_scatter(input_data))
  ✗ features (call bc.features.set_oracle(fn) or bc.features.build_from_data())
```

**Cause:** Trying to use a component before prerequisites are initialized.

**Solution:**
```python
# Correct workflow order:
bc = BundleChoice()

# 1. Config first
bc.load_config(config)

# 2. Then data
bc.data.load_and_scatter(input_data)

# 3. Then features
bc.features.build_from_data()

# 4. Now subproblems work
bc.subproblems.load()
```

**Quick Fix:**
```python
# Use validate_setup to check before solving
bc.validate_setup('row_generation')
```

---

### DimensionMismatchError

**Error Message:**
```
DimensionMismatchError: Dimension mismatch in agent_data['modular']
Mismatches found:
  num_agents: expected 100, got 150
```

**Cause:** Data dimensions don't match configuration.

**Solution:**
```python
# Check your config matches your data
config = {
    "dimensions": {
        "num_agents": 100,  # ← Must match data
        "num_items": 20,    # ← Must match data
        "num_features": 5   # ← Total features
    }
}

# Verify data shapes
print(agent_data['modular'].shape)  # Should be (100, 20, ?)
print(errors.shape)                  # Should be (100, 20) or (simuls, 100, 20)
```

**Quick Fix:**
```python
# Infer dimensions from data
if rank == 0:
    num_agents, num_items = errors.shape[-2:]
    config["dimensions"]["num_agents"] = num_agents
    config["dimensions"]["num_items"] = num_items
```

---

### DataError: Data Contains Invalid Values

**Error Message:**
```
DataError: Data contains invalid values
Problems:
  - agent_data.modular: 15 NaN values
  - errors: 3 Inf values
```

**Cause:** NaN or Inf in input data.

**Common Causes:**
1. Division by zero
2. Log of negative numbers
3. Missing data not handled
4. Overflow in calculations

**Solution:**
```python
# Check for NaN/Inf before loading
import numpy as np

print("NaN count:", np.isnan(data['agent_data']['modular']).sum())
print("Inf count:", np.isinf(data['errors']).sum())

# Fix NaN values
data['agent_data']['modular'] = np.nan_to_num(
    data['agent_data']['modular'],
    nan=0.0  # or use mean imputation
)

# Fix Inf values
data['errors'] = np.clip(data['errors'], -1e10, 1e10)
```

---

### ValidationError: Invalid Quadratic Features

**Error Message:**
```
ValidationError: Invalid quadratic features
  - Quadratic features contain 45 negative values (should be non-negative for supermodularity)
  - Quadratic feature 0 has non-zero diagonal
```

**Cause:** Quadratic features don't meet supermodularity requirements.

**Solution:**
```python
# For supermodular solvers, quadratic features must be:
# 1. Non-negative
quadratic = np.abs(quadratic)  # Make non-negative

# 2. Zero diagonal
for k in range(quadratic.shape[2]):
    np.fill_diagonal(quadratic[:, :, k], 0)

# Verify
from bundlechoice.validation import DataQualityValidator
DataQualityValidator.validate_quadratic_features({"quadratic": quadratic})
```

---

## Setup Issues

### "Unknown subproblem algorithm"

**Error:**
```
ValueError: Unknown subproblem algorithm: 'Gready'  # Typo!
Available algorithms:
  - Greedy
  - LinearKnapsack
  - QuadKnapsack
  - QuadSupermodularNetwork
  - QuadSupermodularLovasz
  - PlainSingleItem
```

**Solution:**
```python
# Check spelling
config = {
    "subproblem": {
        "name": "Greedy"  # ← Correct spelling
    }
}
```

---

### "Gurobi license not found"

**For Row Generation Only** (Ellipsoid doesn't need Gurobi)

**Solutions:**

**Option 1: Use Ellipsoid Method**
```python
# No Gurobi needed
config = {
    "ellipsoid": {"num_iters": 200}
}
theta_hat = bc.ellipsoid.solve()
```

**Option 2: Set Gurobi License**
```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

**Option 3: Academic License**
1. Get free license: https://www.gurobi.com/academia/
2. Run `grbgetkey` with activation code

---

### Feature Oracle Returns Wrong Shape

**Error:**
```
AssertionError: features_oracle must return array of shape (5,). Got (10,) instead.
```

**Cause:** Feature function returns wrong number of features.

**Solution:**
```python
def my_features(agent_id, bundle, data):
    # Count your features!
    linear_feat1 = ...  # 1 feature
    linear_feat2 = ...  # 1 feature  
    quad_feat = ...     # 1 feature
    # Total: 3 features
    
    return np.array([linear_feat1, linear_feat2, quad_feat])

# Config must match
config = {
    "dimensions": {
        "num_features": 3  # ← Must match actual features
    }
}
```

---

## Data Problems

### Estimation Results Are Poor

**Symptoms:**
- Large estimation error
- Theta very far from true values
- Slow convergence

**Diagnostics:**
```python
# 1. Check if subproblem is working
bundles = bc.subproblems.init_and_solve(theta_true)
print("Recovered bundles:", (bundles == obs_bundles).mean())  # Should be ~1.0

# 2. Check feature quality
features_obs = bc.features.compute_gathered_features(obs_bundles)
features_sim = bc.features.compute_gathered_features(bundles)
if rank == 0:
    print("Feature diff:", np.linalg.norm(features_obs - features_sim))
```

**Common Fixes:**

**1. Increase Tolerance:**
```python
config = {
    "row_generation": {
        "max_iters": 200,  # More iterations
        "tolerance_optimality": 1e-8  # Tighter tolerance
    }
}
```

**2. Better Starting Point:**
```python
# Use warm start
theta_hat = bc.row_generation.solve(theta_init=theta_guess)
```

**3. Check Data Quality:**
```python
# Normalize features to similar scales
from sklearn.preprocessing import StandardScaler

agent_data = StandardScaler().fit_transform(
    agent_data.reshape(-1, num_features)
).reshape(num_agents, num_items, num_features)
```

---

### Memory Errors

**Error:**
```
MemoryError: Unable to allocate array with shape (10000, 5000, 100)
```

**Cause:** Data too large for available memory.

**Solutions:**

**1. Reduce Problem Size:**
```python
# Sample agents
sample_idx = np.random.choice(num_agents, size=1000, replace=False)
agent_data = agent_data[sample_idx]
```

**2. Use More MPI Ranks:**
```bash
# Distribute across more processes
mpirun -n 20 python script.py  # Instead of -n 10
```

**3. Reduce Precision:**
```python
# Use float32 instead of float64
agent_data = agent_data.astype(np.float32)
```

**4. Process in Chunks:**
```python
# For very large problems, process agents in batches
for batch in agent_batches:
    bc.data.load_and_scatter(batch)
    results.append(bc.row_generation.solve())
```

---

## Performance Issues

### Slow Estimation

**Diagnostics:**
```python
# Profile subproblem solving
import time

t0 = time.time()
bundles = bc.subproblems.init_and_solve(theta)
t1 = time.time()

if rank == 0:
    print(f"Subproblem time: {t1-t0:.2f}s")
    print(f"Per agent: {(t1-t0)/num_agents*1000:.2f}ms")
```

**Solutions:**

**1. Use Faster Solver:**
```python
# Greedy is much faster than exact methods
config = {"subproblem": {"name": "Greedy"}}  # O(m²)
# vs QuadSupermodular: O(m³)
```

**2. Reduce Iterations:**
```python
config = {
    "row_generation": {
        "max_iters": 50,  # Fewer iterations
        "tolerance_optimality": 0.001  # Looser tolerance
    }
}
```

**3. Optimize Features:**
```python
# Use vectorized features
def vectorized_features(agent_id, bundles, data):
    if bundles.ndim == 2:  # Batch of bundles
        # Compute all at once (faster)
        return batch_compute(bundles)
    else:
        return single_compute(bundles)
```

**4. Enable Caching:**
```python
bc.subproblems.enable_cache()  # Cache repeated solves
```

---

### MPI Communication Bottleneck

**Symptoms:**
- Most time spent in communication
- Poor scaling with more ranks

**Diagnostics:**
```python
from bundlechoice.comm_manager import CommManager

comm_mgr = CommManager(comm, enable_profiling=True)
# ... run estimation ...
profile = comm_mgr.get_comm_profile()

if rank == 0:
    for op, time in profile.items():
        print(f"{op}: {time:.3f}s")
```

**Solutions:**

**1. Use Buffer-Based Communication:**
```python
# Automatic for numpy arrays, but ensure you're not using lists
data = np.array(data)  # Convert to numpy array
```

**2. Reduce Communication Frequency:**
```python
# Solve multiple iterations between communications
# (Advanced: modify solver logic)
```

**3. Better Load Balancing:**
```bash
# Use more balanced rank count
mpirun -n 10 python script.py  # Better than -n 7 for 100 agents
```

---

## MPI Problems

### "Rank 0 hangs, others finish"

**Cause:** Rank 0 doing all the work, others waiting.

**Solution:**
```python
# Ensure all ranks participate in MPI operations
bc.data.load_and_scatter(input_data)  # ALL ranks call this

# Don't do this:
if rank == 0:
    bc.data.load_and_scatter(input_data)  # ❌ Only rank 0
# Other ranks are stuck waiting!
```

---

### "Results differ across runs"

**Cause:** Non-deterministic random number generation.

**Solution:**
```python
# Set seed consistently across all ranks
import numpy as np
np.random.seed(42)  # Before generating data

# MPI-safe seeding
np.random.seed(42 + rank)  # Different seed per rank if needed
```

---

## Solver Failures

### "Greedy returns suboptimal solutions"

**Expected Behavior:** Greedy is an approximation algorithm.

**Improvements:**
```python
# 1. Increase problem quality
# Ensure positive marginal values are added first

# 2. Try different solver
config = {"subproblem": {"name": "LinearKnapsack"}}  # Exact for linear

# 3. Post-process with local search
# (Advanced: implement custom solver with local search)
```

---

### "Row Generation Doesn't Converge"

**Symptoms:**
- Reaches max_iters
- Reduced cost never small enough

**Diagnostics:**
```python
def monitor(info):
    print(f"Iter {info['iteration']}: obj={info['objective']:.4f}")

theta = bc.row_generation.solve(callback=monitor)
```

**Solutions:**

**1. Increase Iterations:**
```python
config = {"row_generation": {"max_iters": 500}}
```

**2. Check Subproblem Quality:**
```python
# Verify subproblem is solving correctly
bundles = bc.subproblems.init_and_solve(theta_test)
# Check if bundles are reasonable
```

**3. Looser Tolerance:**
```python
config = {"row_generation": {"tolerance_optimality": 0.01}}
```

---

## Best Practices

### Before You Start

**1. Validate Your Data:**
```python
from bundlechoice.validation import validate_input_data_comprehensive

validate_input_data_comprehensive(input_data, dimensions_cfg)
```

**2. Check Setup:**
```python
bc.print_status()
bc.validate_setup('row_generation')
```

**3. Test with Small Problem:**
```python
# Start small
test_config = {
    "dimensions": {"num_agents": 10, "num_items": 5, "num_features": 2},
    "subproblem": {"name": "Greedy"},
    "row_generation": {"max_iters": 5}
}

# Verify it works, then scale up
```

---

### During Estimation

**1. Monitor Progress:**
```python
def callback(info):
    if info['iteration'] % 10 == 0:
        print(f"Iteration {info['iteration']}: {info['objective']:.4f}")

theta = bc.row_generation.solve(callback=callback)
```

**2. Save Intermediate Results:**
```python
checkpoints = []

def save_checkpoint(info):
    if info['iteration'] % 20 == 0 and rank == 0:
        np.save(f"checkpoint_{info['iteration']}.npy", info['theta'])

theta = bc.row_generation.solve(callback=save_checkpoint)
```

---

### After Estimation

**1. Validate Results:**
```python
# Check if estimated parameters reproduce observations
bundles_check = bc.subproblems.init_and_solve(theta_hat)

if rank == 0:
    match_rate = (bundles_check == obs_bundles).mean()
    print(f"Match rate: {match_rate*100:.1f}%")
    
    if match_rate < 0.8:
        print("⚠️  Warning: Low match rate. Check:")
        print("  - Solver correctness")
        print("  - Feature specification")
        print("  - Data quality")
```

**2. Analyze Sensitivity:**
```python
# How sensitive are results to perturbations?
theta_perturbed = theta_hat + np.random.normal(0, 0.1, size=num_features)
bundles_pert = bc.subproblems.init_and_solve(theta_perturbed)

if rank == 0:
    sensitivity = (bundles_pert != obs_bundles).mean()
    print(f"Sensitivity: {sensitivity*100:.1f}% change")
```

---

## Getting More Help

**Still stuck?**

1. **Check error message suggestions**: New error system includes fix hints
2. **Run validation**: `bc.validate_setup('row_generation')`
3. **Check examples**: See `examples/` for working code
4. **GitHub Issues**: https://github.com/enzodipasquale/combinatorial-choice-estimation/issues

**Include in bug reports:**
- Error message (full traceback)
- Minimal reproducible example
- Output of `bc.print_status()`
- BundleChoice version
- MPI implementation and version

