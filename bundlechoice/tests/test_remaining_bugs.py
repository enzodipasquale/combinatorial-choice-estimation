#!/usr/bin/env python
"""Test for remaining bugs after user fixes"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_initialize_master_problem_signature():
    """Test that _initialize_master_problem is called with correct arguments"""
    import inspect
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find method definition
    method_def_line = None
    for i, line in enumerate(lines, 1):
        if '_initialize_master_problem(self' in line and 'def' in line:
            method_def_line = i
            break
    
    # Find method call
    call_line = None
    for i, line in enumerate(lines, 1):
        if '_initialize_master_problem(' in line and 'agent_weights' in line:
            call_line = i
            break
    
    if method_def_line and call_line:
        # Check method signature
        def_line = lines[method_def_line - 1]
        call_line_content = lines[call_line - 1]
        
        # Extract parameters from definition
        if 'agent_weights' not in def_line:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"BUG #1: row_generation.py:{call_line}", flush=True)
                print(f"  Method definition (line {method_def_line}): {def_line.strip()}", flush=True)
                print(f"  Method call (line {call_line}): {call_line_content.strip()}", flush=True)
                print(f"  Problem: agent_weights is passed as argument but method doesn't accept it", flush=True)
            return False
    
    return True

def test_base_estimation_cfg():
    """Test that BaseEstimationManager._create_result doesn't use undefined self.cfg"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check if self.cfg is used in _create_result
    in_create_result = False
    bug_line = None
    for i, line in enumerate(lines, 1):
        if 'def _create_result' in line:
            in_create_result = True
        if in_create_result and 'def ' in line and '_create_result' not in line:
            in_create_result = False
        if in_create_result and 'self.cfg.' in line:
            bug_line = i
            break
    
    if bug_line:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"BUG #2: base.py:{bug_line}", flush=True)
            print(f"  Code: {lines[bug_line - 1].strip()}", flush=True)
            print(f"  Problem: BaseEstimationManager doesn't define self.cfg", flush=True)
            print(f"  Fix: Use self.config.row_generation.max_iters or define self.cfg in __init__", flush=True)
        return False
    
    return True

def test_result_type_annotation():
    """Test that EstimationResult has correct type annotation"""
    import os
    
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check for final_objective: None
    for i, line in enumerate(lines, 1):
        if 'final_objective:' in line and ': None' in line and 'Optional' not in line:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"BUG #3: result.py:{i}", flush=True)
                print(f"  Code: {line.strip()}", flush=True)
                print(f"  Problem: Type annotation is 'None' instead of 'Optional[float] = None'", flush=True)
            return False
    
    return True

def test_code_analysis():
    """Static code analysis for remaining bugs"""
    bugs = []
    
    import os
    
    # Bug 1: row_generation.py - agent_weights argument
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    method_def_params = None
    for i, line in enumerate(lines, 1):
        if 'def _initialize_master_problem(self' in line:
            # Extract parameters
            if 'agent_weights' not in line:
                method_def_params = line.strip()
                # Check if it's called with agent_weights
                for j, call_line in enumerate(lines[i:], i+1):
                    if '_initialize_master_problem(' in call_line and 'agent_weights' in call_line:
                        bugs.append({
                            'file': 'row_generation.py',
                            'line': j,
                            'issue': 'agent_weights passed as argument but method signature doesn\'t accept it',
                            'def_line': i,
                            'def': method_def_params
                        })
                        break
    
    # Bug 2: base.py - self.cfg
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'base.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    in_create_result = False
    for i, line in enumerate(lines, 1):
        if 'def _create_result' in line:
            in_create_result = True
        if in_create_result and 'def ' in line and '_create_result' not in line:
            in_create_result = False
        if in_create_result and 'self.cfg.' in line:
            bugs.append({
                'file': 'base.py',
                'line': i,
                'issue': 'self.cfg is not defined in BaseEstimationManager',
                'code': line.strip()
            })
    
    # Bug 3: result.py - type annotation
    file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'result.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if 'final_objective:' in line and ': None' in line and 'Optional' not in line:
            bugs.append({
                'file': 'result.py',
                'line': i,
                'issue': 'Type annotation is None instead of Optional[float] = None',
                'code': line.strip()
            })
    
    if bugs:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n" + "="*60, flush=True)
            print("REMAINING BUGS FOUND:", flush=True)
            print("="*60, flush=True)
            for bug in bugs:
                print(f"\nBug: {bug['file']}:{bug['line']}", flush=True)
                print(f"  Issue: {bug['issue']}", flush=True)
                if 'code' in bug:
                    print(f"  Code: {bug['code']}", flush=True)
                if 'def' in bug:
                    print(f"  Method def (line {bug['def_line']}): {bug['def']}", flush=True)
        return False
    
    return True

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60, flush=True)
        print("Testing for remaining bugs", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('initialize_master_problem_signature', test_initialize_master_problem_signature),
        ('base_estimation_cfg', test_base_estimation_cfg),
        ('result_type_annotation', test_result_type_annotation),
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
