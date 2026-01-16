#!/usr/bin/env python
"""Comprehensive tests for all refactored modules"""
import sys
import os
import importlib.util
import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def import_module_direct(path, module_name):
    """Import module directly"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_code_inspection(file_path, module_name):
    """Inspect code for syntax errors and logical bugs"""
    bugs = []
    
    if not os.path.exists(file_path):
        return bugs
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for incomplete statements
        if stripped and (stripped.endswith('if') or stripped.endswith('for') or stripped.endswith('while') or stripped.endswith('elif') or stripped.endswith('except')):
            bugs.append(f"Line {i}: Incomplete statement: {stripped[:50]}")
        
        # Check for incomplete function definitions
        if stripped.startswith('def ') and ':' not in line:
            bugs.append(f"Line {i}: Incomplete function definition: {stripped[:50]}")
        
        # Check for incomplete property
        if stripped.startswith('@property') and i < len(lines):
            next_line = lines[i].strip() if i < len(lines) else ""
            if next_line and not next_line.startswith('def '):
                bugs.append(f"Line {i}: Property without function definition")
    
    return bugs

def test_row_generation_bugs():
    """Test row_generation.py for bugs"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rg_path = os.path.join(base_path, 'estimation', 'row_generation.py')
    
    bugs = test_code_inspection(rg_path, 'row_generation')
    
    # Read file for specific bug checks
    with open(rg_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check specific bugs
    for i, line in enumerate(lines, 1):
        # Check for undefined variables used before definition
        if 'master_variables = (theta, u)' in line:
            # Check if u is defined before this line
            if i < len(lines):
                next_lines = '\n'.join(lines[i:min(i+5, len(lines))])
                if 'u = ' not in next_lines or next_lines.find('u = ') > next_lines.find('master_variables'):
                    bugs.append(f"Line {i}: Uses 'u' before it's defined in master_variables")
        
        # Check for double self.cfg.self.cfg
        if 'self.cfg.self.cfg' in line:
            bugs.append(f"Line {i}: Double 'self.cfg.self.cfg' - should be 'self.cfg'")
        
        # Check for undefined variables
        if 'self._theta_warmstart' in line and 'self._theta_warmstart =' not in '\n'.join(lines[:i]):
            if 'theta_warmstart' in '\n'.join(lines[:i]) or 'theta_warmstart' in '\n'.join(lines[i:i+10]):
                bugs.append(f"Line {i}: Uses 'self._theta_warmstart' but should be 'theta_warmstart' parameter")
        
        # Check for undefined obj_val
        if 'obj_val' in line and 'obj_val =' not in '\n'.join(lines[:i]):
            if i > 100:  # Only check in solve method area
                bugs.append(f"Line {i}: Uses 'obj_val' but it's not defined")
        
        # Check for wrong method names
        if 'solve_local' in line:
            bugs.append(f"Line {i}: Uses 'solve_local' - check if method exists in subproblem_manager")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ row_generation.py found {len(bugs)} issues:")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ row_generation.py code inspection passed")
    
    return bugs

def test_base_bugs():
    """Test base.py for bugs"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_file = os.path.join(base_path, 'estimation', 'base.py')
    
    bugs = test_code_inspection(base_file, 'base')
    
    # Read file for specific bug checks
    with open(base_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for undefined attributes
    for i, line in enumerate(lines, 1):
        # Check for self.obs_features
        if 'self.obs_features' in line:
            if 'self.obs_features =' not in '\n'.join(lines[:i]):
                bugs.append(f"Line {i}: Uses 'self.obs_features' but it's not defined in __init__")
        
        # Check for self.config.num_obs
        if 'self.config.num_obs' in line:
            bugs.append(f"Line {i}: Uses 'self.config.num_obs' - should be 'self.config.dimensions.num_obs'")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ base.py found {len(bugs)} issues:")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ base.py code inspection passed")
    
    return bugs

def test_oracles_manager_bugs():
    """Test oracles_manager.py for bugs"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    om_path = os.path.join(base_path, 'oracles_manager.py')
    
    bugs = test_code_inspection(om_path, 'oracles_manager')
    
    with open(om_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Check for ignoring _features_oracle_takes_data flag
        if 'self._features_oracle(' in line and 'self.data_manager.local_data' in line:
            # Check if _features_oracle_takes_data is checked
            context_start = max(0, i-10)
            context = '\n'.join(lines[context_start:i+1])
            if '_features_oracle_takes_data' not in context:
                bugs.append(f"Line {i}: features_oracle always passes data parameter, ignores _features_oracle_takes_data flag")
        
        # Check for inconsistent index usage
        if 'self.data_manager.local_id[id]' in line:
            bugs.append(f"Line {i}: Uses 'self.data_manager.local_id[id]' instead of 'id'")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ oracles_manager.py found {len(bugs)} issues:")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ oracles_manager.py code inspection passed")
    
    return bugs

def test_comm_manager():
    """Test comm_manager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cm_path = os.path.join(base_path, 'comm_manager.py')
    cm_module = import_module_direct(cm_path, 'bundlechoice.comm_manager')
    CommManager = cm_module.CommManager
    
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    
    # Test Reduce returns None on non-root
    arr = np.array([1.0, 2.0, 3.0])
    result = cm.Reduce(arr)
    if cm._is_root():
        assert result is not None
    else:
        assert result is None
    
    # Test sum_row_andReduce
    arr2 = np.random.randn(5, 3)
    result2 = cm.sum_row_andReduce(arr2)
    if cm._is_root():
        assert result2 is not None
        assert result2.shape == (3,)
    else:
        assert result2 is None
    
    if cm._is_root():
        print("✓ CommManager methods work correctly")

def test_data_manager():
    """Test data_manager"""
    bugs = test_code_inspection(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_manager.py'),
        'data_manager'
    )
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ data_manager.py found {len(bugs)} issues")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ data_manager.py code inspection passed")

def test_core():
    """Test core.py"""
    bugs = test_code_inspection(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core.py'),
        'core'
    )
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ core.py found {len(bugs)} issues")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ core.py code inspection passed")

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    all_bugs = []
    
    try:
        test_comm_manager()
    except Exception as e:
        all_bugs.append(f"CommManager: {e}")
        if rank == 0:
            import traceback
            traceback.print_exc()
    
    try:
        bugs = test_data_manager()
        all_bugs.extend(bugs or [])
    except Exception as e:
        all_bugs.append(f"DataManager: {e}")
    
    try:
        bugs = test_oracles_manager_bugs()
        all_bugs.extend(bugs or [])
    except Exception as e:
        all_bugs.append(f"OraclesManager: {e}")
    
    try:
        bugs = test_base_bugs()
        all_bugs.extend(bugs or [])
    except Exception as e:
        all_bugs.append(f"Base: {e}")
        if rank == 0:
            import traceback
            traceback.print_exc()
    
    try:
        bugs = test_row_generation_bugs()
        all_bugs.extend(bugs or [])
    except Exception as e:
        all_bugs.append(f"RowGeneration: {e}")
        if rank == 0:
            import traceback
            traceback.print_exc()
    
    try:
        test_core()
    except Exception as e:
        all_bugs.append(f"Core: {e}")
    
    if all_bugs and rank == 0:
        print(f"\n✗ Total {len(all_bugs)} bugs/issues found")
        sys.exit(1)
    elif rank == 0:
        print("\n✓ All code inspections passed!")
