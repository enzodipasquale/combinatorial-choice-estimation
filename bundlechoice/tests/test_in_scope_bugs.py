#!/usr/bin/env python
"""Test for bugs in in-scope modules only"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_core_imports():
    """Test that core.py doesn't import non-existent classes"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'core.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    bugs = []
    
    # Check for ColumnGenerationManager and EllipsoidManager usage
    for i, line in enumerate(lines, 1):
        if 'ColumnGenerationManager' in line or 'EllipsoidManager' in line:
            bugs.append({
                'file': 'core.py',
                'line': i,
                'code': line.strip(),
                'issue': 'Uses ColumnGenerationManager/EllipsoidManager but they are not imported (commented out in __init__.py)'
            })
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("BUGS FOUND in core.py:", flush=True)
            for bug in bugs:
                print(f"  Line {bug['line']}: {bug['code']}", flush=True)
                print(f"    Issue: {bug['issue']}", flush=True)
        return False
    
    return True

def test_result_type_annotation():
    """Test that EstimationResult final_objective type matches usage"""
    import os
    
    result_file = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    base_file = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    
    with open(result_file, 'r') as f:
        result_lines = f.readlines()
    
    with open(base_file, 'r') as f:
        base_content = f.read()
    
    # Check if final_objective can be None
    can_be_none = 'final_objective' in base_content and '= None' in base_content
    
    for i, line in enumerate(result_lines, 1):
        if 'final_objective:' in line:
            if ': None' in line or ': np.float64' in line:
                if can_be_none and 'Optional' not in line:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print(f"BUG: estimation/result.py:{i}", flush=True)
                        print(f"  Code: {line.strip()}", flush=True)
                        print(f"  Issue: final_objective can be None but type annotation doesn't reflect this", flush=True)
                        print(f"  Fix: Change to 'final_objective: Optional[np.float64] = None'", flush=True)
                    return False
            break
    
    return True

def test_solve_subproblems_theta_broadcast():
    """Test if solve_subproblems needs to broadcast theta"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'subproblems', 'subproblem_manager.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check solve_subproblems method
    for i, line in enumerate(lines, 1):
        if 'def solve_subproblems(self, theta):' in line:
            # Check if it broadcasts theta
            method_body = ''.join(lines[i:])
            if 'Bcast' not in method_body and 'theta' in method_body:
                # Check if subproblem.solve() is called directly - it might need broadcasting
                if 'self.subproblem.solve(theta)' in method_body:
                    # Check if initialize_and_solve_subproblems broadcasts
                    init_method = ''.join(lines)
                    if 'initialize_and_solve_subproblems' in init_method:
                        if MPI.COMM_WORLD.Get_rank() == 0:
                            print("INFO: solve_subproblems calls subproblem.solve(theta) directly", flush=True)
                            print("  Note: initialize_and_solve_subproblems broadcasts theta, but solve_subproblems doesn't", flush=True)
                            print("  This might be okay if solve_subproblems is only called after initialization", flush=True)
                    return True
            break
    
    return True

def test_comprehensive_analysis():
    """Comprehensive static analysis of in-scope modules"""
    bugs = []
    import os
    
    # Bug 1: core.py - ColumnGenerationManager and EllipsoidManager
    file_path = os.path.join(os.path.dirname(__file__), '..', 'core.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if 'ColumnGenerationManager' in line or 'EllipsoidManager' in line:
            bugs.append({
                'file': 'core.py',
                'line': i,
                'issue': 'Uses ColumnGenerationManager/EllipsoidManager but they are commented out in __init__.py',
                'code': line.strip()
            })
    
    # Bug 2: result.py - final_objective type
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    base_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    
    with open(file_path, 'r') as f:
        result_lines = f.readlines()
    
    with open(base_path, 'r') as f:
        base_content = f.read()
    
    can_be_none = 'final_objective' in base_content and '= None' in base_content
    
    for i, line in enumerate(result_lines, 1):
        if 'final_objective:' in line:
            if (': None' in line or ': np.float64' in line) and 'Optional' not in line and can_be_none:
                bugs.append({
                    'file': 'estimation/result.py',
                    'line': i,
                    'issue': 'final_objective type annotation doesn\'t allow None but it can be None',
                    'code': line.strip()
                })
            break
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("BUGS FOUND IN IN-SCOPE MODULES:", flush=True)
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
        print("Testing in-scope modules for bugs", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('core_imports', test_core_imports),
        ('result_type_annotation', test_result_type_annotation),
        ('solve_subproblems_theta_broadcast', test_solve_subproblems_theta_broadcast),
        ('comprehensive_analysis', test_comprehensive_analysis),
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
