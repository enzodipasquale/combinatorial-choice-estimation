# Bugs Found in Refactored Modules

## Modules Tested
- `bundlechoice/comm_manager.py` ✓ (fixed)
- `bundlechoice/config.py` ✓
- `bundlechoice/data_manager.py` ✓
- `bundlechoice/oracles_manager.py` ⚠️
- `bundlechoice/core.py` ✓
- `bundlechoice/subproblems/` ✓

## BUGS FOUND

### 1. **Bug in `oracles_manager.features_oracle()` - Ignores `_features_oracle_takes_data` Flag** ⚠️ CRITICAL

**Location:** `bundlechoice/oracles_manager.py:65-71`

**Code:**
```python
def features_oracle(self, bundles, local_id=None):
    if local_id is None:
        local_id = self.data_manager.local_id
    if self._features_oracle_vectorized:
        return self._features_oracle(bundles, local_id, self.data_manager.local_data)  # Always 3 args
    else:
        return np.stack([self._features_oracle(bundles[id], id, self.data_manager.local_data) for id in local_id])  # Always 3 args
```

**Issue:** 
- Always passes 3 arguments `(bundles, local_id, self.data_manager.local_data)` 
- But `_features_oracle_takes_data` flag (set in `set_features_oracle()` line 58) indicates whether oracle expects `data` parameter
- If `_features_oracle_takes_data == False`, oracle only takes 2 args, but code passes 3 → `TypeError`

**Impact:** Will crash when using oracles that don't take `data` parameter (e.g., from `build_local_modular_error_oracle` if used as features oracle).

**Fix:**
```python
def features_oracle(self, bundles, local_id=None):
    if local_id is None:
        local_id = self.data_manager.local_id
    if self._features_oracle_vectorized:
        if self._features_oracle_takes_data:
            return self._features_oracle(bundles, local_id, self.data_manager.local_data)
        else:
            return self._features_oracle(bundles, local_id)
    else:
        if self._features_oracle_takes_data:
            return np.stack([self._features_oracle(bundles[id], id, self.data_manager.local_data) for id in local_id])
        else:
            return np.stack([self._features_oracle(bundles[id], id) for id in local_id])
```

---

### 2. **Bug in `oracles_manager.features_oracle_individual()` - Ignores `_features_oracle_takes_data` Flag** ⚠️ CRITICAL

**Location:** `bundlechoice/oracles_manager.py:73-77`

**Code:**
```python
def features_oracle_individual(self, bundle, local_id):
    if self._features_oracle_vectorized:
        return self._features_oracle(bundle[:, None], local_id, self.data_manager.local_data)  # Always 3 args
    else:
        return self._features_oracle(bundle, local_id, self.data_manager.local_data)  # Always 3 args
```

**Issue:** Always passes 3 arguments, ignores `_features_oracle_takes_data` flag.

**Fix:**
```python
def features_oracle_individual(self, bundle, local_id):
    if self._features_oracle_vectorized:
        bundle_arg = bundle[:, None]
    else:
        bundle_arg = bundle
    if self._features_oracle_takes_data:
        return self._features_oracle(bundle_arg, local_id, self.data_manager.local_data)
    else:
        return self._features_oracle(bundle_arg, local_id)
```

---

### 3. **Bug in `oracles_manager.error_oracle()` - Inconsistent Index Usage** ⚠️

**Location:** `bundlechoice/oracles_manager.py:93-94`

**Code:**
```python
return np.stack([self._error_oracle(bundles[id], 
        self.data_manager.local_id[id]) for id in local_id], )
```

**Issue:** Uses `self.data_manager.local_id[id]` instead of just `id`. Since `id` is already from `local_id` (which is `self.data_manager.local_id` if None), this is redundant and potentially incorrect.

**Fix:**
```python
return np.stack([self._error_oracle(bundles[id], id) for id in local_id])
```

---

### 4. **Potential Issue: `oracles_manager._compute_features_at_obs_bundles()` Returns None on Non-Root**

**Location:** `bundlechoice/oracles_manager.py:31-33`

**Code:**
```python
@lru_cache(maxsize=1)
def _compute_features_at_obs_bundles(self, _version):
    local_obs_features = self.features_oracle(self.data_manager.local_obs_bundles)
    return self.comm_manager.sum_row_andReduce(local_obs_features)
```

**Issue:** After `_Reduce` fix, `sum_row_andReduce()` returns `None` on non-root ranks. The property `_features_at_obs_bundles` will return `None` on non-root.

**Note:** This may be acceptable if property is only accessed on root. Verify usage.

---

### 5. **Potential Issue: `subproblem_manager.initialize_and_solve_subproblems()` - Theta Allocation**

**Location:** `bundlechoice/subproblems/subproblem_manager.py:36-40`

**Code:**
```python
def initialize_and_solve_subproblems(self, theta):
    theta = self.comm_manager.Bcast(theta)
    self.initialize_subproblems()
    local_bundles = self.solve_subproblems(theta)
    return local_bundles
```

**Issue:** `_Bcast` requires `theta` to be allocated numpy array on all ranks. If `theta` is `None` or not allocated on non-root, will crash.

**Note:** Verify calling code ensures `theta` is allocated on all ranks.

---

## Summary

**3 Critical Bugs:**
1. `oracles_manager.features_oracle()` - Ignores `_features_oracle_takes_data` flag (lines 69, 71)
2. `oracles_manager.features_oracle_individual()` - Ignores `_features_oracle_takes_data` flag (lines 75, 77)
3. `oracles_manager.error_oracle()` - Uses `self.data_manager.local_id[id]` instead of `id` (line 94)

**2 Potential Issues:**
- `oracles_manager._compute_features_at_obs_bundles()` returns None on non-root
- `subproblem_manager.initialize_and_solve_subproblems()` may fail if theta not allocated
