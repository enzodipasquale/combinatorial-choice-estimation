# Bug Report - All Refactored Modules

## Summary
**Total: 13 Critical Bugs Found**

---

## base.py (2 bugs)

### Bug #1: Undefined `self.obs_features`
**Lines:** 35, 36, 46, 55
**Issue:** `self.obs_features` is never defined in `__init__` but used in multiple methods.
**Impact:** `AttributeError` when calling `compute_obj_and_grad_at_root()`, `compute_obj()`, or `compute_grad()`.

### Bug #2: Wrong Config Path
**Lines:** 36, 55
**Issue:** Uses `self.config.num_obs` but should be `self.config.dimensions.num_obs`.
**Impact:** `AttributeError` when accessing `self.config.num_obs`.

---

## row_generation.py (9 bugs)

### Bug #1: Uses `u` Before Definition
**Line:** 38
**Issue:** `master_variables = (theta, u)` uses `u` before it's defined on line 41.
**Impact:** `NameError` when initializing master problem.

### Bug #2: Wrong Variable Name
**Line:** 39
**Issue:** Uses `self._theta_warmstart` but parameter is `theta_warmstart`.
**Impact:** Warmstart won't work.

### Bug #3: Wrong Variable Used in Callback
**Line:** 46
**Issue:** Uses `master_variables` which was incorrectly defined on line 38 (before `u` exists).
**Impact:** Callback receives incorrect tuple.

### Bug #4: `u_iter` Not Defined on Non-Root
**Line:** 57
**Issue:** `self.u_iter` is only defined on root (line 52), but `Scatterv_by_row` called on all ranks.
**Impact:** Will fail on non-root when trying to scatter `None`.

### Bug #5: Double `self.cfg.self.cfg`
**Line:** 65
**Issue:** `self.cfg.self.cfg.tol_row_generation` should be `self.cfg.tol_row_generation`.
**Impact:** `AttributeError`.

### Bug #6: Wrong Array Indexing
**Line:** 78
**Issue:** `bundles = self.comm_manager.Gatherv_by_row(features_local[local_violations], ...)` should use `local_pricing_results[local_violations]`.
**Impact:** Gathers wrong data (features instead of bundles).

### Bug #7: Wrong Method Name
**Line:** 114
**Issue:** `solve_local` doesn't exist, should be `solve_subproblems`.
**Impact:** `AttributeError`.

### Bug #8: Wrong Argument Order
**Line:** 107
**Issue:** `_initialize_master_problem(initial_constraints, theta_warmstart, agent_weights)` but method signature doesn't have `agent_weights` parameter.
**Impact:** `TypeError` - too many arguments.

### Bug #9: Undefined `obj_val`
**Line:** 129
**Issue:** `obj_val` is never defined in `solve()` method.
**Impact:** `NameError`.

---

## oracles_manager.py (2 bugs)

### Bug #1: Ignores `_features_oracle_takes_data` Flag
**Lines:** 69, 71, 75, 77
**Issue:** Always passes 3 arguments to features_oracle, ignoring `_features_oracle_takes_data` flag.
**Impact:** `TypeError` when using oracles that don't take `data` parameter.

### Bug #2: Inconsistent Index Usage
**Line:** 94
**Issue:** Uses `self.data_manager.local_id[id]` instead of `id`.
**Impact:** Redundant and potentially incorrect.

---

## Test Results

All tests run with 2-3 MPI processes and 2-3 second timeout wrapper.

**Modules Tested:**
- ✓ comm_manager.py
- ✓ config.py  
- ✓ data_manager.py
- ⚠️ oracles_manager.py (2 bugs)
- ✓ core.py
- ✓ subproblems/
- ⚠️ estimation/base.py (2 bugs)
- ⚠️ estimation/row_generation.py (9 bugs)
