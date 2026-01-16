#!/usr/bin/env python
"""Test for runtime bugs only (no type checking)"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_core_missing_attributes():
    """Test that core.py doesn't access non-existent attributes"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'core.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    bugs = []
    
    # Check for properties that reference commented-out attributes
    for i, line in enumerate(lines, 1):
        # Skip commented-out lines
        if line.strip().startswith('#'):
            continue
        if 'def ellipsoid(self)' in line or 'def column_generation(self)' in line or 'def standard_errors(self)' in line:
            # Check if the corresponding attribute is commented out or doesn't exist
            prop_name = None
            attr_name = None
            if 'ellipsoid(self)' in line:
                prop_name = 'ellipsoid'
                attr_name = 'ellipsoid_manager'
            elif 'column_generation(self)' in line:
                prop_name = 'column_generation'
                attr_name = 'column_generation_manager'
            elif 'standard_errors(self)' in line:
                prop_name = 'standard_errors'
                attr_name = 'standard_errors_manager'
            
            if attr_name:
                # Check if the attribute assignment is commented out
                found_assignment = False
                for j, check_line in enumerate(lines):
                    if f'self.{attr_name} =' in check_line and not check_line.strip().startswith('#'):
                        found_assignment = True
                        break
                
                if not found_assignment:
                    bugs.append({
                        'file': 'core.py',
                        'line': i,
                        'code': line.strip(),
                        'issue': f'Property {prop_name}() accesses self.{attr_name} but attribute is not initialized (commented out)'
                    })
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("BUGS FOUND in core.py:", flush=True)
            for bug in bugs:
                print(f"  Line {bug['line']}: {bug['code']}", flush=True)
                print(f"    Issue: {bug['issue']}", flush=True)
        return False
    
    return True

def test_solve_subproblems_broadcast():
    """Test if solve_subproblems needs theta broadcast"""
    import os
    
    subprob_file = os.path.join(os.path.dirname(__file__), '..', 'subproblems', 'subproblem_manager.py')
    base_file = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    
    with open(subprob_file, 'r') as f:
        subprob_lines = f.readlines()
    
    with open(base_file, 'r') as f:
        base_content = f.read()
    
    # Check if solve_subproblems broadcasts theta
    solve_subproblems_method = None
    for i, line in enumerate(subprob_lines):
        if 'def solve_subproblems(self, theta):' in line:
            method_body = ''.join(subprob_lines[i:i+3])
            if 'Bcast' not in method_body:
                # Check if it's called from base.py
                if 'solve_subproblems(theta)' in base_content:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print("POTENTIAL ISSUE: solve_subproblems doesn't broadcast theta", flush=True)
                        print("  But it might be okay if theta is already synced", flush=True)
    
    return True

def test_comprehensive_runtime_analysis():
    """Comprehensive analysis for runtime bugs only"""
    bugs = []
    import os
    
    # Bug 1: core.py - properties accessing non-existent attributes
    file_path = os.path.join(os.path.dirname(__file__), '..', 'core.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check which managers are initialized
    initialized = {}
    for i, line in enumerate(lines, 1):
        if 'self.' in line and '_manager =' in line:
            if not line.strip().startswith('#'):
                # Extract manager name
                if '=' in line:
                    parts = line.split('=')
                    if len(parts) > 0:
                        lhs = parts[0].strip()
                        if 'self.' in lhs and '_manager' in lhs:
                            attr_name = lhs.split('self.')[1].strip()
                            initialized[attr_name] = True
    
    # Check properties
    for i, line in enumerate(lines, 1):
        # Skip commented-out lines
        if line.strip().startswith('#'):
            continue
        if 'def ' in line and '(self)' in line:
            if 'ellipsoid(self)' in line or 'column_generation(self)' in line or 'standard_errors(self)' in line:
                prop_name = line.split('def ')[1].split('(self)')[0]
                attr_name = f'{prop_name.replace("ellipsoid", "ellipsoid_manager").replace("column_generation", "column_generation_manager").replace("standard_errors", "standard_errors_manager")}'
                
                if attr_name not in initialized:
                    # Check if return statement accesses the attribute
                    for j in range(i, min(i+5, len(lines))):
                        if f'self.{attr_name}' in lines[j]:
                            bugs.append({
                                'file': 'core.py',
                                'line': i,
                                'issue': f'Property {prop_name}() accesses self.{attr_name} but attribute is not initialized',
                                'code': line.strip()
                            })
                            break
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("RUNTIME BUGS FOUND:", flush=True)
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
        print("Testing for runtime bugs only", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('core_missing_attributes', test_core_missing_attributes),
        ('solve_subproblems_broadcast', test_solve_subproblems_broadcast),
        ('comprehensive_runtime_analysis', test_comprehensive_runtime_analysis),
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
