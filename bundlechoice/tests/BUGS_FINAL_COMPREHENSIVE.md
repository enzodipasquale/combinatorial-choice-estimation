# Comprehensive Bug Report - All BundleChoice Modules

## Test Summary
- **Modules Tested:** comm_manager, config, data_manager, oracles_manager, core, subproblems/, estimation/base.py, estimation/row_generation.py
- **Test Method:** Syntax checks, logic checks, runtime tests
- **MPI Processes:** 2-3
- **Timeout:** 2-3 seconds

---

## BUGS FOUND

### 1. **oracles_manager.py - features_oracle_individual ignores flag** ⚠️ CRITICAL

**Lines:** 90, 92

**Code:**
```python
def features_oracle_individual(self, bundle, local_id):
    if self._features_oracle_vectorized:
        return self._features_oracle(bundle[:, None], local_id, self.data_manager.local_data)  # Line 90
    else:
        return self._features_oracle(bundle, local_id, self.data_manager.local_data)  # Line 92
```

**Issue:** Always passes 3 arguments, ignores `_features_oracle_takes_data` flag (same bug as `features_oracle` had).

**Impact:** Will crash with `TypeError` when using oracles that don't take `data` parameter.

**Fix:** Apply same pattern as `features_oracle`:
```python
def features_oracle_individual(self, bundle, local_id):
    data_arg = (self.data_manager.local_data,) if self._features_oracle_takes_data else ()
    bundle_arg = bundle[:, None] if self._features_oracle_vectorized else bundle
    return self._features_oracle(bundle_arg, local_id, *data_arg)
```

---

### 2. **row_generation.py - F-string syntax error** ⚠️ CRITICAL

**Line:** 148

**Code:**
```python
msg = f"Theta hit {bound_type.upper()} bound at indices: {bounds_info[f"hit_{bound_type}"]}"
```

**Issue:** Nested f-string quotes - can't use `f"` inside an f-string with same quote type.

**Impact:** `SyntaxError` - code won't run.

**Fix:** Use different quote type or extract variable:
```python
hit_indices = bounds_info[f"hit_{bound_type}"]
msg = f"Theta hit {bound_type.upper()} bound at indices: {hit_indices}"
```
Or:
```python
msg = f'Theta hit {bound_type.upper()} bound at indices: {bounds_info[f"hit_{bound_type}"]}'
```

---

### 3. **row_generation.py - _create_result missing arguments** ⚠️ CRITICAL

**Line:** 125

**Code:**
```python
result = self._create_result(iteration)
```

**Issue:** `_create_result` signature is:
```python
def _create_result(self, theta, converged, num_iterations, final_objective=None, warnings=None, metadata=None):
```

But call only passes `iteration` (which should be `num_iterations`). Missing `theta` and `converged`.

**Impact:** Will raise `TypeError` - wrong number of arguments.

**Fix:**
```python
result = self._create_result(self.theta_sol, converged, iteration, final_objective=None, warnings=warnings_list)
```

Or calculate `converged` and `final_objective` first.

---

### 4. **row_generation.py - Incomplete function definition (false positive)**

**Line:** 98

**Issue:** Multi-line function definition - this is actually valid Python syntax, but test flagged it.

**Status:** False positive - no fix needed.

---

## Summary

**3 Critical Bugs:**
1. `oracles_manager.features_oracle_individual()` - ignores `_features_oracle_takes_data` flag
2. `row_generation.py` line 148 - f-string syntax error
3. `row_generation.py` line 125 - `_create_result` missing required arguments

**Modules Status:**
- ✓ comm_manager.py
- ✓ config.py
- ✓ data_manager.py
- ⚠️ oracles_manager.py (1 bug)
- ✓ core.py
- ✓ subproblems/
- ✓ estimation/base.py
- ⚠️ estimation/row_generation.py (2 bugs)

---

## Notes

- `features_oracle()` and `error_oracle()` have been fixed to use `data_arg` pattern
- `features_oracle_individual()` still needs the same fix
- `error_oracle_individual()` doesn't need fix (error oracles don't take data)
