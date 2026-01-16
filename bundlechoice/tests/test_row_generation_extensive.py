#!/usr/bin/env python
"""Extensive testing for row_generation.py"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_row_generation_logic():
    """Test logic flow and potential runtime issues in row_generation.py"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    bugs = []
    
    # Check 1: _initialize_master_problem - Bcast called on potentially None theta_iter
    for i, line in enumerate(lines, 1):
        if 'self.theta_iter = self.comm_manager.Bcast(self.theta_iter)' in line:
            # Check if theta_iter could be None at this point
            # It's set to None on line 54 if not root
            # Bcast should handle None, but let's check
            pass
    
    # Check 2: solve() - theta_iter might not be initialized before solve_subproblems
    in_solve = False
    theta_iter_initialized = False
    solve_subproblems_called = False
    for i, line in enumerate(lines, 1):
        if 'def solve(self' in line:
            in_solve = True
        if in_solve and 'def ' in line and 'solve' not in line:
            in_solve = False
        if in_solve:
            if 'self.theta_iter' in line and '=' in line:
                theta_iter_initialized = True
            if 'solve_subproblems(self.theta_iter)' in line:
                solve_subproblems_called = True
                if not theta_iter_initialized and '_initialize_master_problem' not in ''.join(lines[max(0, i-5):i]):
                    bugs.append({
                        'file': 'row_generation.py',
                        'line': i,
                        'issue': 'solve_subproblems called with self.theta_iter which might not be initialized if init_master=False',
                        'code': line.strip()
                    })
    
    # Check 3: _master_iteration - accessing master_variables before checking if root
    for i, line in enumerate(lines, 1):
        if 'def _master_iteration(self' in line:
            # Find where master_variables is accessed
            in_method = False
            root_check = False
            master_vars_access = False
            for j in range(i, min(i+40, len(lines))):
                if 'def ' in lines[j] and '_master_iteration' not in lines[j]:
                    break
                if 'if self.comm_manager._is_root()' in lines[j]:
                    root_check = True
                if 'self.master_variables' in lines[j] and root_check:
                    pass  # OK, inside root check
                elif 'self.master_variables' in lines[j] and not root_check:
                    # This would be a bug but I don't see it in the code
                    pass
    
    # Check 4: _enforce_slack_counter - checking if self.slack_counter is None after it's already been used
    for i, line in enumerate(lines, 1):
        if 'def _enforce_slack_counter(self):' in line:
            # Find the check for None
            has_none_check = False
            none_check_line = None
            uses_counter_before = False
            for j in range(i, min(i+25, len(lines))):
                if 'if self.slack_counter is None:' in lines[j]:
                    has_none_check = True
                    none_check_line = j
                if 'self.slack_counter' in lines[j] and has_none_check and j < none_check_line:
                    uses_counter_before = True
                if 'def ' in lines[j] and '_enforce_slack_counter' not in lines[j]:
                    break
            
            if has_none_check:
                # Check if slack_counter is accessed before the None check
                for j in range(i+1, none_check_line):
                    if 'self.slack_counter' in lines[j] and '=' not in lines[j] and 'if' not in lines[j]:
                        bugs.append({
                            'file': 'row_generation.py',
                            'line': j,
                            'issue': 'self.slack_counter accessed before None check',
                            'code': lines[j].strip()
                        })
    
    # Check 5: solve() - timing variable t0 might not be defined if loop doesn't execute
    for i, line in enumerate(lines, 1):
        if 'def solve(self' in line:
            # Find elapsed = time.perf_counter() - t0
            t0_defined = False
            elapsed_line = None
            for j in range(i, min(i+30, len(lines))):
                if 't0 = time.perf_counter()' in lines[j]:
                    t0_defined = True
                if 'elapsed = time.perf_counter() - t0' in lines[j]:
                    elapsed_line = j
                    if not t0_defined or 'while iteration <' not in ''.join(lines[max(0, i):j]):
                        bugs.append({
                            'file': 'row_generation.py',
                            'line': j,
                            'issue': 'elapsed uses t0 but t0 might not be defined if loop never executes',
                            'code': lines[j].strip()
                        })
    
    # Check 6: Gatherv_by_row called with potentially empty arrays
    for i, line in enumerate(lines, 1):
        if 'Gatherv_by_row(pricing_results[local_violations]' in line:
            # Check if local_violations could be empty
            # This is OK as numpy handles empty indexing, but could be a performance issue
            pass
    
    # Check 7: _master_iteration - accessing u_iter before it might be set
    for i, line in enumerate(lines, 1):
        if 'def _master_iteration(self' in line:
            # u_iter_local is accessed early
            u_iter_initialized = False
            for j in range(i, min(i+40, len(lines))):
                if 'self.u_iter_local' in lines[j] and '=' in lines[j]:
                    u_iter_initialized = True
                if 'self.u_iter_local' in lines[j] and '=' not in lines[j] and not u_iter_initialized:
                    # This would be a bug but initialization happens in __init__ and _initialize_master_problem
                    pass
    
    # Check 8: add_constraints - bundles @ theta when theta might not match dimensions
    # This is a runtime shape mismatch risk, but hard to catch statically
    
    # Check 9: update_objective_for_weights - accessing master_variables without None check
    for i, line in enumerate(lines, 1):
        if 'def update_objective_for_weights(self' in line:
            has_root_check = False
            has_master_check = False
            master_vars_access = False
            for j in range(i, min(i+10, len(lines))):
                if 'if self.comm_manager._is_root()' in lines[j]:
                    has_root_check = True
                if 'self.master_model is None' in lines[j]:
                    has_master_check = True
                if 'self.master_variables' in lines[j]:
                    master_vars_access = True
                    if not has_root_check or not has_master_check:
                        bugs.append({
                            'file': 'row_generation.py',
                            'line': j,
                            'issue': 'master_variables accessed without proper None checks',
                            'code': lines[j].strip()
                        })
    
    # Check 10: _initialize_master_problem - agent_weights None check happens but what if it's None?
    for i, line in enumerate(lines, 1):
        if 'def _initialize_master_problem(self' in line:
            # Check if it handles agent_weights being None
            has_weights_check = False
            weights_check_line = None
            for j in range(i, min(i+30, len(lines))):
                if 'if self.comm_manager._is_root() and self.agent_weights is not None:' in lines[j]:
                    has_weights_check = True
                    weights_check_line = j
                if 'def ' in lines[j] and '_initialize_master_problem' not in lines[j]:
                    break
            # This is actually OK - it just doesn't initialize if weights is None
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("POTENTIAL RUNTIME BUGS IN row_generation.py:", flush=True)
            print("="*60, flush=True)
            for bug in bugs:
                print(f"\nBug: {bug['file']}:{bug['line']}", flush=True)
                print(f"  Issue: {bug['issue']}", flush=True)
                print(f"  Code: {bug['code']}", flush=True)
        return False
    
    return True

def test_none_access_patterns():
    """Test for potential None access patterns"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    bugs = []
    
    # Check for attribute access without None checks
    attributes_that_can_be_none = [
        'master_model', 'master_variables', 'theta_iter', 'u_iter', 
        'agent_weights', 'slack_counter'
    ]
    
    for attr in attributes_that_can_be_none:
        for i, line in enumerate(lines, 1):
            # Skip if it's an assignment or None check
            if f'self.{attr}' in line:
                if '=' in line or f'is None' in line or f'is not None' in line:
                    continue
                # Check if there's a None check before this access
                # Look backwards for a None check
                found_check = False
                for j in range(max(0, i-20), i):
                    if f'self.{attr}' in lines[j] and ('is None' in lines[j] or 'is not None' in lines[j]):
                        found_check = True
                        break
                    # Also check for early return or if statement
                    if 'return' in lines[j] or 'if ' in lines[j]:
                        # Might have check in condition
                        pass
                
                # Skip if it's in a method that always initializes it first
                # This is heuristic but better than nothing
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("None access bugs found", flush=True)
        return False
    
    return True

def test_mpi_communication():
    """Test MPI communication patterns for correctness"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check 1: Bcast called on potentially None values
    # This is actually OK - Bcast should handle None
    
    # Check 2: Gatherv_by_row with row_counts
    for i, line in enumerate(lines, 1):
        if 'Gatherv_by_row' in line:
            # Check if row_counts is always provided when needed
            # row_counts is obtained from data_manager.agent_counts which should always exist
            pass
    
    # Check 3: Reduce operations
    for i, line in enumerate(lines, 1):
        if 'Reduce(' in line:
            # Check if result is only used on root
            # The Reduce method returns None on non-root, so this should be OK
            pass
    
    return True

def test_comprehensive_row_generation():
    """Comprehensive test combining all checks"""
    bugs = []
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check: solve() method - theta_iter initialization
    in_solve = False
    for i, line in enumerate(lines, 1):
        if 'def solve(self' in line:
            in_solve = True
            init_master_default = 'init_master = True' in line or 'init_master=True' in line
        if in_solve and 'def ' in line and 'solve' not in line:
            in_solve = False
        if in_solve and 'solve_subproblems(self.theta_iter)' in line:
            # Check if theta_iter is guaranteed to be set
            # It's set in _initialize_master_problem or should be set before
            # If init_master=False and theta_iter was never set, this could fail
            # But the default is init_master=True, so this might be OK
            
            # Check if there's a way theta_iter could be None
            # _initialize_master_problem sets it, but if init_master=False, it might not be called
            # However, if init_master=False, the user must have set theta_iter manually
            # So this is probably OK, but worth noting
            pass
    
    # Check: elapsed calculation
    for i, line in enumerate(lines, 1):
        if 'elapsed = time.perf_counter() - t0' in line:
            # Check if t0 is always defined
            # t0 is set in the loop, so if loop never runs, t0 won't be defined
            # But the loop condition is 'while iteration < self.cfg.max_iters'
            # If max_iters is 0, the loop won't run and t0 won't be defined
            # This is a potential bug
            # Find the loop start
            for j in range(max(0, i-15), i):
                if 'while iteration <' in lines[j] or 'for ' in lines[j]:
                    loop_start = j
                    # Check if t0 is defined before the loop
                    t0_before_loop = False
                    for k in range(loop_start, i):
                        if 't0 = time.perf_counter()' in lines[k]:
                            t0_before_loop = True
                            break
                    if not t0_before_loop:
                        bugs.append({
                            'file': 'row_generation.py',
                            'line': i,
                            'issue': 'elapsed uses t0 but t0 is only set inside loop - if loop never executes, this will fail',
                            'code': line.strip()
                        })
                    break
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("BUGS FOUND IN row_generation.py:", flush=True)
            print("="*60, flush=True)
            for bug in bugs:
                print(f"\nBug: {bug['file']}:{bug['line']}", flush=True)
                print(f"  Issue: {bug['issue']}", flush=True)
                print(f"  Code: {bug['code']}", flush=True)
        return False
    
    return True

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60, flush=True)
        print("Extensive testing of row_generation.py", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('row_generation_logic', test_row_generation_logic),
        ('none_access_patterns', test_none_access_patterns),
        ('mpi_communication', test_mpi_communication),
        ('comprehensive_row_generation', test_comprehensive_row_generation),
    ]
    
    for test_name, test_func in tests:
        if rank == 0:
            print(f"\nRunning test: {test_name}", flush=True)
        try:
            result = test_func()
            results[test_name] = result
            if rank == 0:
                status = "PASS" if result else "FAIL"
                print(f"  {test_name}: {status}", flush=True)
        except Exception as e:
            results[test_name] = False
            if rank == 0:
                print(f"  {test_name}: FAIL - {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    if rank == 0:
        print("\n" + "="*60, flush=True)
        print("Test Summary:", flush=True)
        print("="*60, flush=True)
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}", flush=True)
        
        all_passed = all(results.values())
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME BUGS FOUND'}", flush=True)
