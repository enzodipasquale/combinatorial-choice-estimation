# Combinatorial Auction Validation

This folder contains scripts for validating the estimation method on the combinatorial auction application.

## Overview

The validation process:
1. **Extract parameters**: Load estimated parameters `theta_hat` from real data estimation
2. **Generate synthetic data**: Use `theta_hat` as `theta_true` to generate synthetic bundles
3. **Run estimation**: Estimate parameters on synthetic data
4. **Compare results**: Compare `theta_hat_synthetic` with `theta_true` to validate the method

## Structure

- `extract_parameters.py` - Extract and save estimated parameters from real data
- `generate_synthetic_data.py` - Generate synthetic bundles using true parameters
- `run_validation.py` - Run estimation on synthetic data and compare results
- `config_small.yaml` - Configuration for small debug settings
- `config.yaml` - Configuration for full validation runs
- `run_validation.sbatch` - Slurm script for running validation
- `run_small.sbatch` - Slurm script for small debug runs

## Usage

### Quick Start (Small Settings for Debugging)

**Using Slurm (Recommended for debugging):**
```bash
sbatch debug.sbatch
```

**Manual steps:**
```bash
# Step 1: Extract parameters from real data estimation
# Note: This will run estimation on the full real data, which may take time
python extract_parameters.py

# Step 2: Generate synthetic data with small dimensions
python generate_synthetic_data.py --config config_small.yaml --seed 42

# Step 3: Run validation on synthetic data
sbatch run_small.sbatch
# Or run directly:
srun ../applications/combinatorial_auction/run-gurobi.bash python run_validation.py --config config_small.yaml
```

### Full Validation Pipeline

For full-scale validation with multiple replications:

```bash
# Step 1: Extract parameters (if not already done)
python extract_parameters.py

# Step 2: Generate synthetic data (single bundle set)
python generate_synthetic_data.py --config config.yaml --seed 42

# Step 3: Run validation (multiple replications)
# Use SLURM job array or loop manually with different --replication-id
for i in {0..9}; do
    srun ../applications/combinatorial_auction/run-gurobi.bash python run_validation.py --replication-id $i
done

# Step 4: Summarize results
python summarize_results.py
```

### Notes

- **Feature Dimensions**: The combinatorial auction uses:
  - Modular agent features: `modular_characteristics_i_j_k.npy` (shape: agents × items × features)
  - Modular item features: Identity matrix (num_items features)
  - Quadratic item features: `quadratic_characteristic_j_j_k.npy` (shape: items × items × features)
  - Total: `num_features = modular_agent_features + num_items + quadratic_features`
  
- For small settings, the scripts automatically subsample theta_true to match the config dimensions.

## Output

Results are saved to:
- `theta_hat_real.npy` - Estimated parameters from real data
- `synthetic_data/` - Directory containing generated synthetic data
- `results/` - Directory containing validation results

