# BundleChoice Feature Proposals

Based on patterns observed in the `combinatorial_auction_v2` application, these features would reduce boilerplate and improve usability.

---

## 1. Feature Naming / Parameter Labels

### Problem

Parameters are currently anonymous (`theta_0`, `theta_1`, ...). Applications must track names separately:

```python
# In compute_se_non_fe.py
PARAM_NAMES = ["bidder_elig_pop", "pop_distance", "travel_survey", "air_travel"]

# Output mapping is manual:
for i, (idx, name) in enumerate(zip(NON_FE_INDICES, PARAM_NAMES)):
    row_data[f"theta_{name}"] = theta_hat[idx]
    row_data[f"se_{name}"] = se_result.se[i]
```

This leads to:
- Unlabeled estimation output
- Manual index-to-name mapping
- Risk of mismatch between indices and names

### Proposed Solution

Add `feature_names` to `DimensionsConfig`:

```yaml
# config.yaml
dimensions:
  num_agents: 256
  num_items: 493
  num_features: 497
  feature_names:
    - bidder_elig_pop      # index 0 (modular agent feature)
    - FE_item_{j}          # indices 1-493 (pattern for fixed effects)
    - pop_distance         # index 494
    - travel_survey        # index 495  
    - air_travel           # index 496
```

Or programmatically:

```python
bc.config.dimensions.set_feature_names([
    "bidder_elig_pop",
    *[f"FE_item_{j}" for j in range(493)],  # Fixed effects
    "pop_distance",
    "travel_survey", 
    "air_travel",
])
```

### Implementation Details

1. Add `feature_names: Optional[List[str]]` to `DimensionsConfig`
2. Add `FeatureManager.get_name(index) -> str` and `get_index(name) -> int`
3. Update `StandardErrorsResult` to include names in output
4. Update estimation result printing to use names when available

### Benefits

- **Readable output**: SE results show `θ[pop_distance] = 552.12` instead of `θ[494] = 552.12`
- **Self-documenting**: Config file serves as feature documentation
- **Less error-prone**: Reference features by name, not index

### Lines Saved

~20 lines per application + significantly improved UX

---

## 2. Data Loading from Directory

### Problem

Every application repeats ~50 lines of data loading boilerplate:

```python
# run_estimation.py lines 67-112
INPUT_DIR = get_input_dir(DELTA, WINNERS_ONLY, HQ_DISTANCE)

with open(os.path.join(INPUT_DIR, "metadata.json"), "r") as f:
    input_metadata = json.load(f)

obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

item_data = {
    "modular": -np.eye(num_items),
    "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
    "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
}
agent_data = {
    "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
    "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
}

input_data = {
    "item_data": item_data,
    "agent_data": agent_data,
    "errors": errors,
    "obs_bundle": obs_bundle
}
```

### Proposed Solution

Add `DataManager.load_from_directory(path)`:

```python
# Before (50+ lines)
# ... all the boilerplate above ...

# After (1 line)
bc.data.load_from_directory("input_data/delta4/")
```

### Convention-Based File Detection

The method would auto-detect files based on naming conventions:

| File Pattern | Maps To |
|--------------|---------|
| `metadata.json` | Dimensions + feature names |
| `matching_*.npy` or `obs_bundle*.npy` | `obs_bundle` |
| `modular_*_i_j_k.npy` | `agent_data["modular"]` |
| `modular_*_j_k.npy` or `modular_*_j.npy` | `item_data["modular"]` |
| `quadratic_*_j_j_k.npy` | `item_data["quadratic"]` |
| `quadratic_*_i_j_j_k.npy` | `agent_data["quadratic"]` |
| `capacity_*.npy` | `agent_data["capacity"]` |
| `weight_*.npy` | `item_data["weights"]` |

### Implementation Details

```python
class DataManager:
    def load_from_directory(
        self, 
        path: str,
        generate_errors: bool = True,
        error_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load input data from a directory following BundleChoice conventions.
        
        Args:
            path: Directory containing .npy files and metadata.json
            generate_errors: If True, generate errors from metadata
            error_seed: Random seed for error generation
            
        Returns:
            Loaded input_data dictionary
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Update dimensions from metadata
            self._update_dimensions_from_metadata(metadata)
        
        # Auto-detect and load arrays
        input_data = self._load_arrays_from_directory(path)
        
        # Generate errors if requested
        if generate_errors:
            input_data["errors"] = self._generate_errors(error_seed)
        
        return input_data
```

### Benefits

- **DRY**: Eliminates repeated loading code across applications
- **Convention over configuration**: Standard directory structure is self-describing
- **Error reduction**: No manual dimension counting or file path typos

### Lines Saved

~50 lines per application

---

## 3. Parameter Bounds by Name

### Problem

Setting bounds by index is error-prone and requires mental index tracking:

```python
# run_estimation.py lines 140-155
theta_lbs = np.zeros(num_features)
theta_ubs = np.full(num_features, 1000.0)

# theta[0] >= 75 (modular parameter)
theta_lbs[0] = 75

# theta[-3] between 400 and 650 (pop/distance)
theta_lbs[-3] = 400
theta_ubs[-3] = 650

# theta[-2] >= -120 (travel survey)
theta_lbs[-2] = -120

# theta[-1] >= -75 (air travel)
theta_lbs[-1] = -75
```

Comments are required to understand what each index means.

### Proposed Solution

Allow bounds by feature name (requires Feature Naming to be implemented first):

```python
# Clear and self-documenting
bc.bounds.set("bidder_elig_pop", lower=75)
bc.bounds.set("pop_distance", lower=400, upper=650)
bc.bounds.set("travel_survey", lower=-120)
bc.bounds.set("air_travel", lower=-75)
```

Or via config:

```yaml
row_generation:
  bounds:
    bidder_elig_pop:
      lower: 75
    pop_distance:
      lower: 400
      upper: 650
    travel_survey:
      lower: -120
    air_travel:
      lower: -75
```

### Implementation Details

```python
class BoundsManager:
    """Manages parameter bounds with name-based access."""
    
    def __init__(self, dimensions_cfg: DimensionsConfig):
        self.dimensions_cfg = dimensions_cfg
        self._lower = np.full(dimensions_cfg.num_features, -np.inf)
        self._upper = np.full(dimensions_cfg.num_features, np.inf)
    
    def set(
        self, 
        name_or_index: Union[str, int],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        """Set bounds for a parameter by name or index."""
        if isinstance(name_or_index, str):
            idx = self.dimensions_cfg.get_feature_index(name_or_index)
        else:
            idx = name_or_index
            
        if lower is not None:
            self._lower[idx] = lower
        if upper is not None:
            self._upper[idx] = upper
    
    def set_pattern(
        self,
        pattern: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        """Set bounds for all features matching a pattern (e.g., 'FE_item_*')."""
        import fnmatch
        for i, name in enumerate(self.dimensions_cfg.feature_names):
            if fnmatch.fnmatch(name, pattern):
                self.set(i, lower=lower, upper=upper)
```

### Benefits

- **Self-documenting**: Code reads like English
- **No index errors**: Can't accidentally set wrong parameter
- **Pattern matching**: Set bounds for all fixed effects at once with `FE_item_*`

### Lines Saved

~5 lines per application, but significantly improved readability and reduced bugs

---

## Implementation Priority

| Priority | Feature | Complexity | Dependency |
|----------|---------|------------|------------|
| 🔴 **1st** | Feature Naming | Low | None |
| 🔴 **2nd** | Data Loading from Directory | Medium | None |
| 🟡 **3rd** | Parameter Bounds by Name | Low | Feature Naming |

Feature Naming should be implemented first as Parameter Bounds by Name depends on it.

---

## Next Steps

1. Review and discuss these proposals
2. Decide on API design details
3. Implement in order of priority
4. Update `combinatorial_auction_v2` application to use new features
5. Document in library README
