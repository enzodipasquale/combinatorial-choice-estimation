#!/usr/bin/env python
"""Comprehensive debugging tests for all bundlechoice modules"""
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

def test_code_syntax(file_path, module_name):
    """Check for syntax errors"""
    bugs = []
    if not os.path.exists(file_path):
        return bugs
    
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
    except SyntaxError as e:
        bugs.append(f"SyntaxError: {e.msg} at line {e.lineno}")
    except Exception as e:
        bugs.append(f"Error reading file: {e}")
    
    return bugs

def test_code_logic(file_path, module_name):
    """Check for logical bugs"""
    bugs = []
    if not os.path.exists(file_path):
        return bugs
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Incomplete statements
        if stripped and (stripped.endswith('if') or stripped.endswith('for') or stripped.endswith('while') or 
                        stripped.endswith('elif') or stripped.endswith('except') or stripped.endswith('with')):
            bugs.append(f"Line {i}: Incomplete statement: {stripped[:60]}")
        
        # Incomplete function definitions (but allow multi-line)
        if stripped.startswith('def ') and ':' not in line:
            # Check if next line continues
            if i < len(lines) and not lines[i].strip().startswith((' ', '\t')):
                bugs.append(f"Line {i}: Incomplete function definition: {stripped[:60]}")
        
        # Double self.cfg.self.cfg
        if 'self.cfg.self.cfg' in line:
            bugs.append(f"Line {i}: Double 'self.cfg.self.cfg'")
    
    return bugs

def test_comm_manager():
    """Test CommManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cm_path = os.path.join(base_path, 'comm_manager.py')
    
    bugs = test_code_syntax(cm_path, 'comm_manager')
    bugs.extend(test_code_logic(cm_path, 'comm_manager'))
    
    try:
        cm_module = import_module_direct(cm_path, 'bundlechoice.comm_manager')
        CommManager = cm_module.CommManager
        comm = MPI.COMM_WORLD
        cm = CommManager(comm)
        
        # Test Reduce
        arr = np.array([1.0, 2.0, 3.0])
        result = cm.Reduce(arr)
        if cm._is_root():
            assert result is not None, "Reduce should return result on root"
        else:
            assert result is None, "Reduce should return None on non-root"
        
        # Test sum_row_andReduce
        arr2 = np.random.randn(5, 3)
        result2 = cm.sum_row_andReduce(arr2)
        if cm._is_root():
            assert result2 is not None and result2.shape == (3,), "sum_row_andReduce shape wrong"
        else:
            assert result2 is None, "sum_row_andReduce should return None on non-root"
        
    except Exception as e:
        bugs.append(f"Runtime error: {e}")
        if MPI.COMM_WORLD.Get_rank() == 0:
            import traceback
            traceback.print_exc()
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ comm_manager.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ comm_manager.py: OK")
    
    return bugs

def test_config():
    """Test config"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, 'config.py')
    
    bugs = test_code_syntax(config_path, 'config')
    bugs.extend(test_code_logic(config_path, 'config'))
    
    try:
        config_module = import_module_direct(config_path, 'bundlechoice.config')
        BundleChoiceConfig = config_module.BundleChoiceConfig
        cfg = BundleChoiceConfig()
        assert cfg.dimensions is not None
        assert cfg.subproblem is not None
    except Exception as e:
        bugs.append(f"Runtime error: {e}")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ config.py: {len(bugs)} issues")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ config.py: OK")
    
    return bugs

def test_data_manager():
    """Test DataManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dm_path = os.path.join(base_path, 'data_manager.py')
    
    bugs = test_code_syntax(dm_path, 'data_manager')
    bugs.extend(test_code_logic(dm_path, 'data_manager'))
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ data_manager.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ data_manager.py: OK")
    
    return bugs

def test_oracles_manager():
    """Test OraclesManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    om_path = os.path.join(base_path, 'oracles_manager.py')
    
    bugs = test_code_syntax(om_path, 'oracles_manager')
    bugs.extend(test_code_logic(om_path, 'oracles_manager'))
    
    with open(om_path, 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
    
    # Check for _features_oracle_takes_data flag usage
    for i, line in enumerate(lines, 1):
        # Check features_oracle - should use data_arg pattern
        if i >= 65 and i <= 72 and 'self._features_oracle(' in line:
            context = '\n'.join(lines[max(0, i-3):min(len(lines), i+1)])
            if 'data_arg' not in context and 'self.data_manager.local_data' in line:
                bugs.append(f"Line {i}: features_oracle doesn't use data_arg pattern for _features_oracle_takes_data flag")
        
        # Check features_oracle_individual
        if i >= 88 and i <= 92 and 'self._features_oracle(' in line:
            context = '\n'.join(lines[max(0, i-3):min(len(lines), i+1)])
            if '_features_oracle_takes_data' not in context and 'self.data_manager.local_data' in line:
                bugs.append(f"Line {i}: features_oracle_individual doesn't check _features_oracle_takes_data flag")
        
        # Check error_oracle index bug
        if 'self.data_manager.local_id[id]' in line:
            bugs.append(f"Line {i}: Uses 'self.data_manager.local_id[id]' instead of 'id'")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ oracles_manager.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ oracles_manager.py: OK")
    
    return bugs

def test_core():
    """Test core"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_path = os.path.join(base_path, 'core.py')
    
    bugs = test_code_syntax(core_path, 'core')
    bugs.extend(test_code_logic(core_path, 'core'))
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ core.py: {len(bugs)} issues")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ core.py: OK")
    
    return bugs

def test_subproblems():
    """Test subproblems module"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subproblems_dir = os.path.join(base_path, 'subproblems')
    
    bugs = []
    for root, dirs, files in os.walk(subproblems_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_path)
                file_bugs = test_code_syntax(file_path, rel_path)
                file_bugs.extend(test_code_logic(file_path, rel_path))
                bugs.extend([f"{rel_path}: {b}" for b in file_bugs])
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ subproblems/: {len(bugs)} issues")
        for bug in bugs[:10]:  # Limit output
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ subproblems/: OK")
    
    return bugs

def test_base():
    """Test estimation/base.py"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_file = os.path.join(base_path, 'estimation', 'base.py')
    
    bugs = test_code_syntax(base_file, 'base')
    bugs.extend(test_code_logic(base_file, 'base'))
    
    with open(base_file, 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
    
    # Check for undefined self.obs_features
    init_content = content[:content.find('def compute_obj')] if 'def compute_obj' in content else content
    for i, line in enumerate(lines, 1):
        if 'self.obs_features' in line:
            if 'self.obs_features =' not in init_content:
                bugs.append(f"Line {i}: Uses 'self.obs_features' but it's not defined in __init__")
        
        if 'self.config.num_obs' in line:
            bugs.append(f"Line {i}: Uses 'self.config.num_obs' but should be 'self.config.dimensions.num_obs'")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ estimation/base.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ estimation/base.py: OK")
    
    return bugs

def test_row_generation():
    """Test estimation/row_generation.py"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rg_path = os.path.join(base_path, 'estimation', 'row_generation.py')
    
    bugs = test_code_syntax(rg_path, 'row_generation')
    bugs.extend(test_code_logic(rg_path, 'row_generation'))
    
    with open(rg_path, 'r') as f:
        lines = f.readlines()
        content = '\n'.join(lines)
    
    # Check specific bugs
    for i, line in enumerate(lines, 1):
        # Check for f-string syntax error
        if 'f"' in line and 'bounds_info[f"' in line:
            # Check for nested quotes issue
            if line.count('f"') > 1 or (line.count('"') - line.count('\\"')) % 2 != 0:
                bugs.append(f"Line {i}: F-string syntax error - nested quotes issue")
        
        # Check for wrong argument order in _initialize_master_problem call
        if '_initialize_master_problem(' in line and 'agent_weights' in line:
            # Check method signature
            for j in range(max(0, i-50), i):
                if 'def _initialize_master_problem' in lines[j]:
                    sig_line = lines[j]
                    # Check if agent_weights is in signature
                    sig_end = j
                    for k in range(j, min(j+5, len(lines))):
                        if '):' in lines[k] or (k > j and lines[k].strip() and not lines[k].strip().startswith((' ', '\t'))):
                            sig_end = k
                            break
                    sig_content = '\n'.join(lines[j:sig_end+1])
                    if 'agent_weights' not in sig_content:
                        bugs.append(f"Line {i}: Passes 'agent_weights' to _initialize_master_problem but method doesn't accept it")
                    break
        
        # Check for undefined obj_val
        if 'obj_val' in line and i > 100 and 'obj_val =' not in '\n'.join(lines[:i]):
            if 'def ' not in line:  # Not a function definition
                bugs.append(f"Line {i}: Uses 'obj_val' but it's not defined")
        
        # Check for wrong method name
        if 'solve_local' in line:
            bugs.append(f"Line {i}: Uses 'solve_local' - check if method exists")
        
        # Check _create_result call
        if '_create_result(' in line:
            # Check if all required args are provided
            if 'self.theta_sol' not in '\n'.join(lines[max(0, i-3):i+1]) and 'theta' not in line:
                bugs.append(f"Line {i}: _create_result call may be missing required arguments")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ estimation/row_generation.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ estimation/row_generation.py: OK")
    
    return bugs

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    all_bugs = []
    
    print(f"\n{'='*60}")
    print(f"Comprehensive Debug Test - Rank {rank}")
    print(f"{'='*60}\n")
    
    all_bugs.extend(test_comm_manager())
    all_bugs.extend(test_config())
    all_bugs.extend(test_data_manager())
    all_bugs.extend(test_oracles_manager())
    all_bugs.extend(test_core())
    all_bugs.extend(test_subproblems())
    all_bugs.extend(test_base())
    all_bugs.extend(test_row_generation())
    
    if all_bugs and rank == 0:
        print(f"\n{'='*60}")
        print(f"✗ TOTAL: {len(all_bugs)} bugs/issues found")
        print(f"{'='*60}")
        sys.exit(1)
    elif rank == 0:
        print(f"\n{'='*60}")
        print("✓ All modules passed!")
        print(f"{'='*60}")
