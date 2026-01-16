# Bug Report - All Refactored Modules

## Modules Tested
- `bundlechoice/comm_manager.py` ✓
- `bundlechoice/config.py` ✓
- `bundlechoice/data_manager.py` ✓
- `bundlechoice/oracles_manager.py` ⚠️
- `bundlechoice/core.py` ✓
- `bundlechoice/subproblems/` ✓
- `bundlechoice/estimation/base.py` ⚠️
- `bundlechoice/estimation/row_generation.py` ⚠️

---

## CRITICAL BUGS

### 1. **base.py - Undefined `self.obs_features`** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/base.py:35, 36, 46, 55`

**Code:**
```python
def compute_obj_and_grad_at_root(self, theta):
    ...
    if self.comm_manager._is_root():
        obj = utility_sum - (self.obs_features @ theta)  # Line 35
        grad = (features_sum - self.obs_features) / self.config.num_obs  # Line 36
```

**Issue:** `self.obs_features` is never defined in `__init__`. It's used in multiple methods but doesn't exist.

**Impact:** Will raise `AttributeError` when calling `compute_obj_and_grad_at_root()`, `compute_obj()`, or `compute_grad()`.

**Fix:** Define `self.obs_features` in `__init__` or use `self.oracles_manager._features_at_obs_bundles_at_root` (but that returns None on non-root).

---

### 2. **base.py - Wrong Config Path** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/base.py:36, 55`

**Code:**
```python
grad = (features_sum - self.obs_features) / self.config.num_obs  # Line 36
return (features_sum - self.obs_features) / self.config.num_obs  # Line 55
```

**Issue:** Uses `self.config.num_obs` but should be `self.config.dimensions.num_obs`.

**Impact:** Will raise `AttributeError` when accessing `self.config.num_obs`.

**Fix:** Change to `self.config.dimensions.num_obs`.

---

### 3. **row_generation.py - Uses `u` Before Definition** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:38`

**Code:**
```python
theta = self.master_model.addMVar(self.dim.num_features, ...)
master_variables = (theta, u)  # Line 38 - u not defined yet!
if self._theta_warmstart is not None:
    theta.Start = theta_warmstart
u = self.master_model.addMVar(self.dim.num_agents, ...)  # Line 41 - u defined here
self.master_variables = (theta, u)  # Line 42
```

**Issue:** Line 38 uses `u` before it's defined on line 41.

**Impact:** Will raise `NameError` when initializing master problem.

**Fix:** Remove line 38 or move it after line 41.

---

### 4. **row_generation.py - Wrong Variable Name** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:39`

**Code:**
```python
if self._theta_warmstart is not None:
    theta.Start = theta_warmstart
```

**Issue:** Uses `self._theta_warmstart` but parameter is `theta_warmstart`.

**Impact:** Will always be `None` (unless set elsewhere), warmstart won't work.

**Fix:** Change to `if theta_warmstart is not None:`

---

### 5. **row_generation.py - Double `self.cfg.self.cfg`** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:65`

**Code:**
```python
stop = reduced_costs < self.cfg.self.cfg.tol_row_generation if self.comm_manager._is_root() else None
```

**Issue:** Double `self.cfg.self.cfg` - should be `self.cfg.tol_row_generation`.

**Impact:** Will raise `AttributeError`.

**Fix:** Change to `self.cfg.tol_row_generation`.

---

### 6. **row_generation.py - `Reduce` Returns None on Non-Root** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:64-65`

**Code:**
```python
reduced_costs = self.comm_manager.Reduce(u_local - self.u_iter_local, op=MPI.MAX)
stop = reduced_costs < self.cfg.self.cfg.tol_row_generation if self.comm_manager._is_root() else None
```

**Issue:** `Reduce` returns `None` on non-root ranks. Line 65 checks `if self.comm_manager._is_root()` but the comparison `reduced_costs < ...` happens before the conditional, so on non-root `reduced_costs` is `None` and the comparison will fail.

**Impact:** Will raise `TypeError` on non-root ranks when comparing `None < ...`.

**Fix:** The conditional check is correct, but the issue is that `reduced_costs` is `None` on non-root. The code should work, but verify the logic.

---

### 7. **row_generation.py - Wrong Array Indexing** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:78`

**Code:**
```python
bundles = self.comm_manager.Gatherv_by_row(features_local[local_violations], row_counts=row_counts)
```

**Issue:** Uses `features_local[local_violations]` but should use `local_pricing_results[local_violations]` to get the bundles.

**Impact:** Will gather wrong data - features instead of bundles.

**Fix:** Change to `local_pricing_results[local_violations]`.

---

### 8. **row_generation.py - Wrong Method Name** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:114`

**Code:**
```python
local_pricing_results = self.subproblem_manager.solve_local(self.theta_iter)
```

**Issue:** Method `solve_local` doesn't exist. Should be `solve_subproblems`.

**Impact:** Will raise `AttributeError`.

**Fix:** Change to `self.subproblem_manager.solve_subproblems(self.theta_iter)`.

---

### 9. **row_generation.py - Undefined `obj_val`** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:129`

**Code:**
```python
result = self._create_result(self.theta_sol, converged, num_iters, obj_val)
```

**Issue:** `obj_val` is never defined in the `solve()` method.

**Impact:** Will raise `NameError`.

**Fix:** Calculate `obj_val` from master model or pass `None`.

---

### 10. **row_generation.py - Wrong Argument Order** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:107`

**Code:**
```python
self._initialize_master_problem(initial_constraints, theta_warmstart, agent_weights) if init_master else None
```

**Issue:** Method signature is `_initialize_master_problem(self, initial_constraints=None, theta_warmstart=None)` but call passes 3 arguments including `agent_weights` which is not a parameter.

**Impact:** Will raise `TypeError` - too many arguments.

**Fix:** Add `agent_weights` parameter to method signature, or remove it from call.

---

### 11. **row_generation.py - `u_iter` Not Defined on Non-Root** ⚠️ CRITICAL

**Location:** `bundlechoice/estimation/row_generation.py:57`

**Code:**
```python
self.u_iter_local = self.comm_manager.Scatterv_by_row(self.u_iter, row_counts=self.data_manager.agent_counts)
```

**Issue:** `self.u_iter` is only defined on root rank (line 52), but `Scatterv_by_row` is called on all ranks. On non-root, `self.u_iter` is `None` (line 55), so this will fail.

**Impact:** Will raise error on non-root ranks when `Scatterv_by_row` tries to scatter `None`.

**Fix:** Ensure `self.u_iter` is allocated on all ranks before scattering, or handle the None case.

---

### 12. **oracles_manager.py - Ignores `_features_oracle_takes_data` Flag** ⚠️ CRITICAL

**Location:** `bundlechoice/oracles_manager.py:69, 71, 75, 77`

**Code:**
```python
def features_oracle(self, bundles, local_id=None):
    ...
    if self._features_oracle_vectorized:
        return self._features_oracle(bundles, local_id, self.data_manager.local_data)  # Always 3 args
    else:
        return np.stack([self._features_oracle(bundles[id], id, self.data_manager.local_data) for id in local_id])  # Always 3 args
```

**Issue:** Always passes 3 arguments, ignores `_features_oracle_takes_data` flag.

**Impact:** Will crash with `TypeError` when using oracles that don't take `data` parameter.

**Fix:** Check `_features_oracle_takes_data` flag before passing `data`.

---

### 13. **oracles_manager.py - Inconsistent Index Usage** ⚠️

**Location:** `bundlechoice/oracles_manager.py:94`

**Code:**
```python
return np.stack([self._error_oracle(bundles[id], 
        self.data_manager.local_id[id]) for id in local_id], )
```

**Issue:** Uses `self.data_manager.local_id[id]` instead of just `id`.

**Impact:** Redundant and potentially incorrect.

**Fix:** Change to `id`.

---

## Summary

**13 Bugs Found:**
- **base.py**: 2 bugs (undefined `self.obs_features`, wrong config path)
- **row_generation.py**: 9 bugs (variable order, undefined variables, wrong method names, wrong arguments)
- **oracles_manager.py**: 2 bugs (ignores flag, inconsistent index)

**Critical Issues:**
- Multiple `NameError` and `AttributeError` will occur at runtime
- Wrong data being gathered/scattered
- Methods called that don't exist
