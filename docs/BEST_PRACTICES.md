# BundleChoice Best Practices

## Table of Contents
1. [Code Organization](#code-organization)
2. [Data Management](#data-management)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling](#error-handling)
5. [Testing](#testing)
6. [Production Deployment](#production-deployment)

---

## Code Organization

### Project Structure

**Recommended layout:**
```
my_project/
├── config/
│   ├── base.yaml          # Base configuration
│   ├── production.yaml    # Production settings
│   └── development.yaml   # Dev/testing settings
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned data
│   └── input/            # BundleChoice input format
├── src/
│   ├── data_prep.py      # Data preparation
│   ├── features.py       # Feature engineering
│   ├── estimation.py     # Estimation logic
│   └── analysis.py       # Results analysis
├── notebooks/
│   └── exploration.ipynb # Data exploration
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_estimation.py
└── run_estimation.py     # Main entry point
```

---

### Separate Configuration from Code

**❌ Bad:**
```python
# Hardcoded values scattered in code
bc = BundleChoice()
bc.load_config({
    "dimensions": {"num_agents": 100, "num_items": 20, "num_features": 5},
    "row_generation": {"max_iters": 50, "tolerance_optimality": 0.001}
})
```

**✅ Good:**
```python
# config/production.yaml
dimensions:
  num_agents: 100
  num_items: 20
  num_features: 5

subproblem:
  name: QuadSupermodularNetwork

row_generation:
  max_iters: 200
  tolerance_optimality: 1e-8
  gurobi_settings:
    OutputFlag: 0
    Threads: 8

# run_estimation.py
import os
bc = BundleChoice()
bc.load_config(os.getenv('CONFIG_FILE', 'config/production.yaml'))
```

---

### Modular Feature Engineering

**❌ Bad: Monolithic function**
```python
def features(agent_id, bundle, data):
    # 100 lines of complex calculations
    # Hard to test, debug, or modify
    ...
```

**✅ Good: Modular components**
```python
# features.py
def compute_distance_features(agent_id, bundle, data):
    """Distance-based features."""
    distances = data['item_data']['distance']
    return distances @ bundle

def compute_size_penalty(bundle):
    """Bundle size penalty."""
    return -np.sum(bundle) ** 2

def features_oracle(agent_id, bundle, data):
    """Combine feature components."""
    dist_feat = compute_distance_features(agent_id, bundle, data)
    size_feat = compute_size_penalty(bundle)
    return np.array([dist_feat, size_feat])

# Easy to test each component
def test_distance_features():
    bundle = np.array([1, 0, 1])
    result = compute_distance_features(0, bundle, test_data)
    assert result.shape == (2,)
```

---

## Data Management

### Data Validation Pipeline

**Always validate data before estimation:**

```python
# data_prep.py
from bundlechoice.validation import validate_input_data_comprehensive
from bundlechoice.errors import ValidationError

def prepare_input_data(raw_data):
    """Clean and validate data."""
    
    # 1. Clean data
    data = clean_missing_values(raw_data)
    data = normalize_features(data)
    
    # 2. Create input format
    input_data = {
        "agent_data": {"modular": data['agent_features']},
        "item_data": {"modular": data['item_features']},
        "errors": generate_errors(data)
    }
    
    # 3. Validate
    try:
        validate_input_data_comprehensive(input_data, dimensions_cfg)
        print("✓ Data validation passed")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
        raise
    
    return input_data
```

---

### Handle Missing Data Properly

**❌ Bad: Ignore missing data**
```python
# NaN values silently propagate
agent_data = raw_data['features']  # Contains NaN
```

**✅ Good: Explicit handling**
```python
def handle_missing_data(data, strategy='mean'):
    """Handle missing values explicitly."""
    if strategy == 'mean':
        # Mean imputation
        col_means = np.nanmean(data, axis=0)
        idx = np.where(np.isnan(data))
        data[idx] = np.take(col_means, idx[1])
    elif strategy == 'drop':
        # Drop rows with missing values
        data = data[~np.isnan(data).any(axis=1)]
    elif strategy == 'zero':
        # Fill with zeros
        data = np.nan_to_num(data, nan=0.0)
    
    # Verify no NaN remain
    assert not np.isnan(data).any(), "NaN values remain after handling"
    
    return data
```

---

### Save Intermediate Results

**Save processed data for reproducibility:**

```python
# After expensive data preparation
np.savez_compressed(
    'data/processed/input_data.npz',
    agent_modular=input_data['agent_data']['modular'],
    item_modular=input_data['item_data']['modular'],
    errors=input_data['errors'],
    obs_bundle=input_data['obs_bundle']
)

# Later, quick reload
def load_processed_data():
    data = np.load('data/processed/input_data.npz')
    return {
        "agent_data": {"modular": data['agent_modular']},
        "item_data": {"modular": data['item_modular']},
        "errors": data['errors'],
        "obs_bundle": data['obs_bundle']
    }
```

---

## Performance Optimization

### Choose the Right Solver

**Decision tree:**

```python
def choose_solver(problem_characteristics):
    """Select optimal solver based on problem."""
    
    if problem_characteristics['has_capacity_constraint']:
        if problem_characteristics['is_quadratic']:
            return "QuadKnapsack"
        else:
            return "LinearKnapsack"
    
    if problem_characteristics['is_supermodular']:
        if problem_characteristics['num_items'] < 100:
            return "QuadSupermodularLovasz"  # Faster for small
        else:
            return "QuadSupermodularNetwork"  # Scales better
    
    # Default: Fast greedy
    return "Greedy"
```

---

### Optimize Feature Computation

**1. Vectorize when possible:**

```python
# ❌ Slow: Loop over bundles
def features_slow(agent_id, bundles, data):
    return np.array([
        compute_single(agent_id, bundle, data) 
        for bundle in bundles.T
    ])

# ✅ Fast: Vectorized
def features_fast(agent_id, bundles, data):
    if bundles.ndim == 1:
        return compute_single(agent_id, bundles, data)
    else:
        # Process all bundles at once
        return data['agent_data']['modular'][agent_id] @ bundles
```

**2. Precompute when possible:**

```python
class PrecomputedFeatures:
    """Cache expensive computations."""
    
    def __init__(self, data):
        # Precompute agent-specific terms
        self.agent_terms = [
            self._precompute_agent(i, data) 
            for i in range(data['num_agents'])
        ]
    
    def __call__(self, agent_id, bundle, data):
        # Use precomputed values
        return self.agent_terms[agent_id] @ bundle

features = PrecomputedFeatures(data)
bc.features.set_oracle(features)
```

---

### MPI Best Practices

**1. Minimize rank 0 bottlenecks:**

```python
# ❌ Bad: All processing on rank 0
if rank == 0:
    results = expensive_computation(all_data)
    processed = post_process(results)

# ✅ Good: Distributed processing
local_results = expensive_computation(local_data)
all_results = comm.gather(local_results, root=0)

if rank == 0:
    processed = post_process(all_results)
```

**2. Use appropriate communication:**

```python
# For small data: use bcast
config = comm.bcast(config, root=0)

# For large arrays: use buffer-based methods
large_array = comm_manager.broadcast_array(large_array, root=0)

# For very large data: use scatter
local_chunk = comm_manager.scatter_array(huge_array, root=0)
```

---

### Profile and Optimize

**1. Profile estimation:**

```python
import cProfile
import pstats

def profile_estimation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    theta_hat = bc.row_generation.solve()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Run on rank 0 only
if rank == 0:
    profile_estimation()
```

**2. Profile MPI communication:**

```python
comm_manager = CommManager(comm, enable_profiling=True)

# ... run estimation ...

if rank == 0:
    profile = comm_manager.get_comm_profile()
    for op, time in sorted(profile.items(), key=lambda x: x[1], reverse=True):
        print(f"{op:30s}: {time:6.3f}s")
```

---

## Error Handling

### Graceful Degradation

**Handle errors gracefully:**

```python
def robust_estimation(bc, theta_init=None):
    """Estimate with fallback strategies."""
    
    # Try row generation first
    try:
        theta = bc.row_generation.solve(theta_init=theta_init)
        return theta, 'row_generation'
    except Exception as e:
        logger.warning(f"Row generation failed: {e}")
    
    # Fallback to ellipsoid
    try:
        theta = bc.ellipsoid.solve()
        return theta, 'ellipsoid'
    except Exception as e:
        logger.error(f"All methods failed: {e}")
        raise
```

---

### Informative Error Messages

**Provide context in custom errors:**

```python
from bundlechoice.errors import DataError

def load_data_with_validation(filepath):
    """Load and validate data with helpful errors."""
    
    try:
        data = np.load(filepath)
    except FileNotFoundError:
        raise DataError(
            message=f"Data file not found: {filepath}",
            suggestion=(
                f"Please ensure the file exists:\n"
                f"  - Check path: {filepath}\n"
                f"  - Run data preparation: python prepare_data.py"
            ),
            context={'filepath': filepath}
        )
    
    # Validate
    if 'errors' not in data:
        raise DataError(
            message="Required key 'errors' missing from data file",
            suggestion="Regenerate data with: python prepare_data.py --include-errors",
            context={'available_keys': list(data.keys())}
        )
    
    return data
```

---

## Testing

### Test Data Preparation

```python
# tests/test_data.py
import pytest
import numpy as np
from src.data_prep import prepare_input_data

def test_data_preparation():
    """Test data preparation pipeline."""
    raw_data = {
        'agent_features': np.random.randn(100, 20, 5),
        'errors': np.random.randn(100, 20)
    }
    
    input_data = prepare_input_data(raw_data)
    
    # Verify structure
    assert 'agent_data' in input_data
    assert 'errors' in input_data
    
    # Verify no NaN
    assert not np.isnan(input_data['agent_data']['modular']).any()
    assert not np.isnan(input_data['errors']).any()
```

---

### Test Features

```python
# tests/test_features.py
def test_feature_determinism():
    """Features should be deterministic."""
    bundle = np.random.choice([0, 1], size=20)
    
    feat1 = my_features(0, bundle, test_data)
    feat2 = my_features(0, bundle, test_data)
    
    assert np.allclose(feat1, feat2)

def test_feature_vectorization():
    """Vectorized features should match loop."""
    bundles = np.random.choice([0, 1], size=(20, 10))
    
    # Vectorized
    feat_vec = my_features(0, bundles, test_data)
    
    # Loop
    feat_loop = np.array([
        my_features(0, bundles[:, i], test_data)
        for i in range(10)
    ]).T
    
    assert np.allclose(feat_vec, feat_loop)
```

---

### Integration Tests

```python
# tests/test_estimation.py
@pytest.mark.mpi(min_size=4)
def test_end_to_end_estimation():
    """Test complete estimation workflow."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Generate test problem
    theta_true = np.array([1.0, 0.5, 1.2])
    if rank == 0:
        input_data = generate_test_data(theta_true)
    else:
        input_data = None
    
    # Estimate
    bc = BundleChoice().quick_setup(config, input_data)
    obs_bundles = bc.generate_observations(theta_true)
    theta_hat = bc.row_generation.solve()
    
    # Verify
    if rank == 0:
        error = np.linalg.norm(theta_hat - theta_true)
        assert error < 0.5, f"Estimation error too large: {error}"
```

---

## Production Deployment

### Configuration Management

**Use environment-specific configs:**

```python
# run_estimation.py
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    
    # Load config
    bc = BundleChoice()
    bc.load_config(args.config)
    
    # Load data
    input_data = load_data(args.data) if rank == 0 else None
    bc.data.load_and_scatter(input_data)
    
    # Run estimation
    theta_hat = bc.row_generation.solve()
    
    # Save results
    if rank == 0:
        np.save(f"{args.output}/theta_hat.npy", theta_hat)

if __name__ == '__main__':
    main()
```

**Run with:**
```bash
mpirun -n 20 python run_estimation.py \
    --config config/production.yaml \
    --data data/processed/input_data.npz \
    --output results/run_001
```

---

### Logging

**Set up proper logging:**

```python
# setup_logging.py
import logging
from bundlechoice.utils import get_logger

def setup_logging(rank, level=logging.INFO):
    """Configure logging for MPI application."""
    
    # File handler for all ranks
    handler = logging.FileHandler(f'logs/rank_{rank}.log')
    handler.setFormatter(
        logging.Formatter('[%(asctime)s][Rank %(rank)d][%(name)s] %(message)s')
    )
    
    # Only rank 0 logs to console
    if rank == 0:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(console)
    
    logging.basicConfig(level=level, handlers=[handler])

# In main script
from mpi4py import MPI
setup_logging(MPI.COMM_WORLD.Get_rank())
```

---

### Checkpointing

**Save progress for long runs:**

```python
def estimation_with_checkpoints(bc, checkpoint_dir='checkpoints'):
    """Estimate with automatic checkpointing."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    results = []
    
    def save_checkpoint(info):
        if info['iteration'] % 50 == 0 and rank == 0:
            checkpoint = {
                'iteration': info['iteration'],
                'theta': info['theta'],
                'objective': info['objective'],
                'timestamp': datetime.now().isoformat()
            }
            filename = f"{checkpoint_dir}/checkpoint_{info['iteration']:04d}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint, f)
            results.append(checkpoint)
            print(f"✓ Saved checkpoint: {filename}")
    
    theta_hat = bc.row_generation.solve(callback=save_checkpoint)
    
    # Save final result
    if rank == 0:
        with open(f"{checkpoint_dir}/final_result.pkl", 'wb') as f:
            pickle.dump({'theta': theta_hat, 'checkpoints': results}, f)
    
    return theta_hat
```

---

### Cluster Deployment

**SLURM job script:**

```bash
#!/bin/bash
#SBATCH --job-name=bundlechoice_estimation
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

module load python/3.9
module load openmpi/4.1.1
module load gurobi/11.0

# Set Gurobi license
export GRB_LICENSE_FILE=$HOME/gurobi.lic

# Run estimation
mpirun -n 100 python run_estimation.py \
    --config config/cluster.yaml \
    --data /scratch/user/data/input_data.npz \
    --output /scratch/user/results/run_${SLURM_JOB_ID}

# Copy results back
cp -r /scratch/user/results/run_${SLURM_JOB_ID} $HOME/results/
```

---

## Summary Checklist

**Before Estimation:**
- [ ] Validate data with `validate_input_data_comprehensive()`
- [ ] Check setup with `bc.print_status()`
- [ ] Test with small problem first
- [ ] Set up logging

**During Development:**
- [ ] Use modular feature engineering
- [ ] Write unit tests for each component
- [ ] Profile performance bottlenecks
- [ ] Use version control for configs

**For Production:**
- [ ] Use environment-specific configs
- [ ] Implement checkpointing for long runs
- [ ] Set up proper logging
- [ ] Test on cluster before full run
- [ ] Save intermediate results

**After Estimation:**
- [ ] Validate results (match rate, sensitivity)
- [ ] Save all parameters and configurations
- [ ] Document any issues or deviations
- [ ] Archive results with metadata

