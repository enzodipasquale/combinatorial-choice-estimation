# Factory Validation

The `validation.py` script verifies that factory-generated data matches manual generation for all scenarios when using the same seed.

## Usage

Run validation for all factories:

```bash
python -m bundlechoice.scenarios.validation
```

Or with a custom seed:

```bash
python -m bundlechoice.scenarios.validation 123
```

## Adding New Factories

When adding a new factory scenario:

1. Add a validation function following the pattern:
   ```python
   def validate_new_scenario(seed: int) -> Tuple[bool, str]:
       # Manual generation code
       # Factory generation code
       # Compare and return (True/False, message)
   ```

2. Add it to the `VALIDATORS` dictionary:
   ```python
   VALIDATORS = {
       ...
       "new_scenario": validate_new_scenario,
   }
   ```

3. Run the validation to ensure it passes.

## What It Validates

For each scenario, the validation script:
- Generates data manually (using `np.random.default_rng(seed)`)
- Generates data using the factory (with the same seed)
- Compares all generated arrays (agent data, item data, errors, weights, capacities, etc.)
- Reports any mismatches

All comparisons use `np.allclose` with `rtol=1e-10` for floating-point arrays, and `np.array_equal` for integer arrays.

