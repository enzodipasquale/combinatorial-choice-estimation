# Debugging Prompt for AI Assistant

## CRITICAL: Where to Work

**YOU MUST ONLY CREATE AND EDIT FILES IN:**
- `bundlechoice/tests/` directory

**DO NOT EDIT ANY FILES OUTSIDE OF `bundlechoice/tests/`**

You can READ files from the codebase to understand them, but ALL your work (test files, bug reports, etc.) must be in the `bundlechoice/tests/` folder.

---

## Scope: What to Debug

### ✅ INCLUDE These Modules:

**Main bundlechoice modules:**
- `bundlechoice/comm_manager.py`
- `bundlechoice/config.py`
- `bundlechoice/core.py`
- `bundlechoice/data_manager.py`
- `bundlechoice/oracles_manager.py`
- `bundlechoice/subproblems/` (entire subproblems module including registry)

**Estimation modules (ONLY these 3):**
- `bundlechoice/estimation/base.py`
- `bundlechoice/estimation/row_generation.py`
- `bundlechoice/estimation/result.py`

### ❌ EXCLUDE These:

- `bundlechoice/scenarios/` - DO NOT debug this
- `bundlechoice/estimation/standard_errors/` - DO NOT debug this
- `bundlechoice/estimation/ellipsoid.py` - DO NOT debug this
- `bundlechoice/estimation/column_generation.py` - DO NOT debug this
- Any other estimation modules not listed above

---

## Testing Requirements

### MPI Testing:
- Always use MPI with 2-3 processes: `mpirun -n 2` or `mpirun -n 3`
- Always use timeout wrapper: `./run_with_timeout.sh <seconds> mpirun -n <procs> python ...`
- Use smallest timeout possible (2-5 seconds for syntax/logic tests, longer for runtime tests)
- Activate `.bundle` venv before running if needed

### Test Structure:
1. **Syntax checks** - Verify Python syntax is valid
2. **Logic checks** - Look for:
   - Undefined variables
   - Missing imports
   - Wrong method names
   - Missing None checks
   - Incomplete statements
   - Type errors
3. **Runtime tests** - Test actual functionality where possible
4. **Code analysis** - Static analysis for potential bugs

---

## Bug Reporting Format

**IMPORTANT: Report bugs DIRECTLY IN CHAT, NOT in markdown files.**

Use this exact format for each bug:

```
### Bug #N: [Module] - [Brief Description]

**Location:** [file_path:line_number]

**Code snippet:**
```python
[relevant code showing the bug]
```

**Problem:**
- [Explanation of what's wrong]
- [Why it's a bug]
- [What will happen if not fixed]

**Impact:** [Runtime error type or behavior issue]

**Fix:** [Suggested fix code or approach]
```

**DO NOT create .md files for bug reports. Report directly in chat.**

---

## Key Things to Check

### 1. Import Errors
- Missing imports (e.g., `Any` from `typing`)
- Circular imports
- Wrong import paths

### 2. Method/Attribute Errors
- Calling methods that don't exist
- Accessing attributes before initialization
- Wrong method names (e.g., `solve()` vs `solve_subproblems()`)

### 3. None Checks
- Accessing attributes on `None` objects
- Methods that should check for `None` but don't

### 4. Type Errors
- Wrong argument types
- Missing required arguments
- Wrong number of arguments

### 5. Logic Errors
- Variables used before definition
- Wrong variable names
- Incomplete statements

### 6. MPI-Specific Issues
- Operations that should only run on root
- Operations that need data on all ranks
- Broadcast/scatter/gather errors

---

## Example Test Structure

Create test files in `bundlechoice/tests/` like:

```python
#!/usr/bin/env python
"""Test for [module name]"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Import modules directly to avoid circular imports
def test_module():
    # Test code here
    pass

if __name__ == '__main__':
    # Run tests with MPI
    pass
```

---

## Specific Areas to Focus On

### Subproblems Module:
- Test all subproblems in registry (Greedy, LinearKnapsack, QuadKnapsack, etc.)
- Test `SubproblemManager` methods: `load()`, `initialize_subproblems()`, `solve_subproblems()`
- Check for None checks before calling methods
- Verify method names are correct

### Row Generation:
- Test `RowGenerationManager.solve()` method
- Check `_initialize_master_problem()` arguments
- Verify `_master_iteration()` logic
- Check `_create_result()` calls have correct arguments
- Test interaction with `SubproblemManager`

### Base Estimation:
- Check `BaseEstimationManager` initialization
- Verify `compute_obj()`, `compute_grad()` methods
- Check for undefined attributes (e.g., `self.obs_features`)

### Result:
- Verify `EstimationResult` dataclass
- Check all type annotations are valid
- No missing imports

---

## Workflow

1. **Read** the target module files to understand structure
2. **Create** test files in `bundlechoice/tests/`
3. **Run** tests with MPI and timeout wrapper
4. **Report** bugs in chat using the format above
5. **Continue** until all modules are tested

---

## Important Notes

- You can READ any file in the codebase
- You can ONLY WRITE/EDIT files in `bundlechoice/tests/`
- Report bugs in chat, not in files
- Use MPI for all tests (2-3 processes)
- Always use timeout wrapper
- Be thorough - test edge cases, None values, wrong arguments, etc.

---

## Current Known Issues (Check if Fixed)

1. `result.py` - Missing `Any` import (should be fixed by removing typing)
2. `subproblem_manager.py` - `solve_subproblems()` doesn't check if `subproblem` is None
3. `row_generation.py` - Calls `subproblem_manager.solve_subproblems()` but should be `solve_subproblems()`

Verify these are fixed and find any additional bugs.

---

## Start Here

Begin by:
1. Creating a comprehensive test file in `bundlechoice/tests/test_extensive_debug.py`
2. Testing all modules in scope
3. Reporting all bugs found in chat using the format above
4. Being thorough and systematic

Good luck!
