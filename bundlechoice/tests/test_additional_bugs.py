#!/usr/bin/env python
"""Test for additional bugs found"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_solve_subproblems_local():
    """Test if solve_subproblems_local method exists"""
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
    
    if hasattr(SubproblemManager, 'solve_subproblems_local'):
        return True
    
    # Check if it's called
    import os
    files_to_check = [
        '../estimation/standard_errors/sandwich.py',
        '../estimation/column_generation.py',
        '../estimation/row_generation_1slack.py',
    ]
    
    bugs = []
    for rel_path in files_to_check:
        file_path = os.path.join(os.path.dirname(__file__), rel_path)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if 'solve_subproblems_local' in line:
                    bugs.append(f"{os.path.basename(file_path)}:{i}")
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("BUG: solve_subproblems_local is called but doesn't exist in SubproblemManager", flush=True)
            for bug in bugs:
                print(f"  Called at: {bug}", flush=True)
        return False
    
    return True

def test_final_objective_type():
    """Test that final_objective type annotation matches usage"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check type annotation
    for i, line in enumerate(lines, 1):
        if 'final_objective:' in line:
            if ': np.float64' in line and 'Optional' not in line:
                # Check if it's set to None anywhere
                base_path = os.path.join(os.path.dirname(__file__), '..', 'estimation')
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path2 = os.path.join(root, file)
                            with open(file_path2, 'r') as f2:
                                content = f2.read()
                                if 'final_objective' in content and '= None' in content:
                                    if MPI.COMM_WORLD.Get_rank() == 0:
                                        print(f"BUG: result.py:{i}", flush=True)
                                        print(f"  Type annotation: {line.strip()}", flush=True)
                                        print(f"  Problem: final_objective can be None but type is np.float64", flush=True)
                                        print(f"  Fix: Change to Optional[np.float64] = None", flush=True)
                                    return False
            break
    
    return True

def test_create_result_signatures():
    """Test that all _create_result calls have correct signatures"""
    import os
    import re
    
    base_path = os.path.join(os.path.dirname(__file__), '..', 'estimation')
    bugs = []
    
    # Get the signature from base.py
    base_file = os.path.join(base_path, 'base.py')
    with open(base_file, 'r') as f:
        base_lines = f.readlines()
    
    # Find _create_result definition
    sig_params = None
    for i, line in enumerate(base_lines, 1):
        if 'def _create_result(self' in line:
            # Extract parameters
            match = re.search(r'def _create_result\(self, ([^)]+)\)', line)
            if match:
                sig_params = match.group(1).split(',')
                sig_params = [p.strip().split('=')[0].strip() for p in sig_params]
                break
    
    if not sig_params:
        return True  # Can't check without signature
    
    # Check all calls
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if '_create_result(' in line and 'def ' not in line:
                        # Count arguments
                        # Simple check: count commas in the call
                        call_match = re.search(r'_create_result\(([^)]+)\)', line)
                        if call_match:
                            args = call_match.group(1).split(',')
                            # Remove self from count
                            num_args = len([a for a in args if a.strip()])
                            expected = len(sig_params)  # Already excludes self
                            if num_args != expected:
                                rel_path = os.path.relpath(file_path, base_path)
                                bugs.append({
                                    'file': rel_path,
                                    'line': i,
                                    'code': line.strip(),
                                    'expected': expected,
                                    'got': num_args
                                })
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("BUG: _create_result called with wrong number of arguments", flush=True)
            for bug in bugs:
                print(f"  {bug['file']}:{bug['line']}", flush=True)
                print(f"    Code: {bug['code']}", flush=True)
                print(f"    Expected {bug['expected']} args, got {bug['got']}", flush=True)
        return False
    
    return True

def test_base_compute_obj():
    """Test that compute_obj method is complete"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    in_compute_obj = False
    has_return = False
    for i, line in enumerate(lines, 1):
        if 'def compute_obj(self' in line:
            in_compute_obj = True
        if in_compute_obj and 'def ' in line and 'compute_obj' not in line:
            if not has_return:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"BUG: base.py compute_obj method may be incomplete", flush=True)
                return False
            break
        if in_compute_obj and 'return' in line:
            has_return = True
    
    return True

def test_code_analysis():
    """Comprehensive code analysis"""
    bugs = []
    
    import os
    
    # Bug 1: solve_subproblems_local
    files_to_check = [
        ('estimation/standard_errors/sandwich.py', 216),
        ('estimation/column_generation.py', 89),
        ('estimation/row_generation_1slack.py', 97),
    ]
    
    for rel_path, line_num in files_to_check:
        file_path = os.path.join(os.path.dirname(__file__), '..', rel_path)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            if line_num <= len(lines) and 'solve_subproblems_local' in lines[line_num - 1]:
                bugs.append({
                    'file': rel_path,
                    'line': line_num,
                    'issue': 'solve_subproblems_local called but method doesn\'t exist in SubproblemManager'
                })
    
    # Bug 2: final_objective type
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'final_objective:' in line and ': np.float64' in line and 'Optional' not in line:
            # Check if it's set to None
            base_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
            with open(base_path, 'r') as f2:
                if 'final_objective' in f2.read() and '= None' in f2.read():
                    bugs.append({
                        'file': 'estimation/result.py',
                        'line': i,
                        'issue': 'final_objective type is np.float64 but can be None',
                        'code': line.strip()
                    })
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("ADDITIONAL BUGS FOUND:", flush=True)
            print("="*60, flush=True)
            for bug in bugs:
                print(f"\nBug: {bug['file']}:{bug['line']}", flush=True)
                print(f"  Issue: {bug['issue']}", flush=True)
                if 'code' in bug:
                    print(f"  Code: {bug['code']}", flush=True)
        return False
    
    return True

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60, flush=True)
        print("Testing for additional bugs", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('solve_subproblems_local', test_solve_subproblems_local),
        ('final_objective_type', test_final_objective_type),
        ('create_result_signatures', test_create_result_signatures),
        ('base_compute_obj', test_base_compute_obj),
        ('code_analysis', test_code_analysis),
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
