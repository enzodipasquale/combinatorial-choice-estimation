#!/usr/bin/env python
"""Comprehensive test for debugging bundlechoice modules"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.config import BundleChoiceConfig, DimensionsConfig
        from bundlechoice.core import BundleChoice
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        from bundlechoice.subproblems.subproblem_base import BaseSubproblem
        from bundlechoice.subproblems.subproblem_registry import SUBPROBLEM_REGISTRY
        from bundlechoice.estimation.base import BaseEstimationManager
        from bundlechoice.estimation.row_generation import RowGenerationManager
        from bundlechoice.estimation.result import EstimationResult
        return True
    except Exception as e:
        print(f"Import error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_comm_manager():
    """Test CommManager basic functionality"""
    try:
        comm = MPI.COMM_WORLD
        cm = CommManager(comm)
        assert cm.rank is not None
        assert cm.comm_size is not None
        assert cm.root == 0
        return True
    except Exception as e:
        print(f"CommManager error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test Config classes"""
    try:
        cfg = BundleChoiceConfig()
        assert cfg.dimensions is not None
        assert cfg.subproblem is not None
        assert cfg.row_generation is not None
        return True
    except Exception as e:
        print(f"Config error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_result_dataclass():
    """Test EstimationResult dataclass"""
    try:
        from bundlechoice.estimation.result import EstimationResult
        theta = np.array([1.0, 2.0, 3.0])
        result = EstimationResult(
            theta_hat=theta,
            converged=True,
            num_iterations=10,
            final_objective=1.5
        )
        assert result.theta_hat is not None
        assert result.converged is True
        return True
    except Exception as e:
        print(f"EstimationResult error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_subproblem_manager_methods():
    """Test SubproblemManager has required methods"""
    try:
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        
        comm = MPI.COMM_WORLD
        cfg = BundleChoiceConfig()
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        # Check methods exist
        assert hasattr(sm, 'load')
        assert hasattr(sm, 'initialize_subproblems')
        assert hasattr(sm, 'initialize_and_solve_subproblems')
        
        # Check that solve() method doesn't exist (it should be on subproblem, not manager)
        # Actually, let me check what methods are expected
        return True
    except Exception as e:
        print(f"SubproblemManager methods error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_base_estimation_manager():
    """Test BaseEstimationManager for method calls"""
    try:
        from bundlechoice.estimation.base import BaseEstimationManager
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        
        comm = MPI.COMM_WORLD
        cfg = BundleChoiceConfig()
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        bem = BaseEstimationManager(cm, cfg, dm, om, sm)
        
        # Check that subproblem_manager has solve method or not
        # This will reveal if solve() is called incorrectly
        if hasattr(sm, 'solve'):
            print("WARNING: SubproblemManager has solve() method", flush=True)
        else:
            print("INFO: SubproblemManager does NOT have solve() method", flush=True)
        
        # Check cfg attribute
        if hasattr(bem, 'cfg'):
            print("WARNING: BaseEstimationManager has cfg attribute", flush=True)
        else:
            print("INFO: BaseEstimationManager does NOT have cfg attribute", flush=True)
        
        return True
    except Exception as e:
        print(f"BaseEstimationManager error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_row_generation_manager():
    """Test RowGenerationManager for method signature issues"""
    try:
        from bundlechoice.estimation.row_generation import RowGenerationManager
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        
        comm = MPI.COMM_WORLD
        cfg = BundleChoiceConfig()
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        
        # Check _initialize_master_problem signature
        import inspect
        sig = inspect.signature(rgm._initialize_master_problem)
        params = list(sig.parameters.keys())
        print(f"_initialize_master_problem parameters: {params}", flush=True)
        
        # Check if subproblem_manager has solve method
        if hasattr(sm, 'solve'):
            print("WARNING: SubproblemManager has solve() method", flush=True)
        else:
            print("INFO: SubproblemManager does NOT have solve() method", flush=True)
        
        return True
    except Exception as e:
        print(f"RowGenerationManager error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_oracles_manager_none_check():
    """Test OraclesManager for None comparison bug"""
    try:
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        
        comm = MPI.COMM_WORLD
        cfg = BundleChoiceConfig()
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        
        # Check the utility_oracle method for None comparison
        import inspect
        import ast
        import os
        
        # Read the source file to check for the bug
        file_path = os.path.join(os.path.dirname(__file__), '..', 'oracles_manager.py')
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Check for the bug: type(local_id) == None
        if 'type(local_id) == None' in source:
            print("BUG FOUND: oracles_manager.py uses 'type(local_id) == None' instead of 'local_id is None'", flush=True)
            return False
        
        return True
    except Exception as e:
        print(f"OraclesManager None check error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def test_code_analysis():
    """Static code analysis for common bugs"""
    bugs_found = []
    
    try:
        import os
        import ast
        
        base_path = os.path.join(os.path.dirname(__file__), '..')
        
        # Check oracles_manager.py
        file_path = os.path.join(base_path, 'oracles_manager.py')
        with open(file_path, 'r') as f:
            source = f.read()
            lines = source.split('\n')
        
        for i, line in enumerate(lines, 1):
            if 'type(' in line and '== None' in line:
                bugs_found.append(f"oracles_manager.py:{i} - Uses 'type(...) == None' instead of 'is None'")
        
        # Check row_generation.py for method call issues
        file_path = os.path.join(base_path, 'estimation', 'row_generation.py')
        with open(file_path, 'r') as f:
            source = f.read()
            lines = source.split('\n')
        
        for i, line in enumerate(lines, 1):
            if 'subproblem_manager.solve_subproblems(' in line:
                bugs_found.append(f"row_generation.py:{i} - Calls subproblem_manager.solve_subproblems() which doesn't exist")
            if '_initialize_master_problem(' in line and 'agent_weights' in line:
                # Check if agent_weights is passed as argument
                bugs_found.append(f"row_generation.py:{i} - _initialize_master_problem called with agent_weights but method doesn't accept it")
        
        # Check base.py
        file_path = os.path.join(base_path, 'estimation', 'base.py')
        with open(file_path, 'r') as f:
            source = f.read()
            lines = source.split('\n')
        
        for i, line in enumerate(lines, 1):
            if 'subproblem_manager.solve_subproblems(' in line:
                bugs_found.append(f"base.py:{i} - Calls subproblem_manager.solve_subproblems() which doesn't exist")
            if 'self.cfg.' in line and 'BaseEstimationManager' in source.split('\n')[0:20]:
                # Check if self.cfg is used but not defined
                bugs_found.append(f"base.py:{i} - Uses self.cfg but it's not defined in BaseEstimationManager")
        
        # Check result.py
        file_path = os.path.join(base_path, 'estimation', 'result.py')
        with open(file_path, 'r') as f:
            source = f.read()
            lines = source.split('\n')
        
        for i, line in enumerate(lines, 1):
            if 'final_objective: None' in line:
                bugs_found.append(f"result.py:{i} - final_objective type annotation is 'None' instead of Optional[float]")
        
        if bugs_found:
            print("STATIC ANALYSIS BUGS FOUND:", flush=True)
            for bug in bugs_found:
                print(f"  - {bug}", flush=True)
            return False
        
        return True
    except Exception as e:
        print(f"Code analysis error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60, flush=True)
        print("Starting comprehensive debug tests", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('imports', test_imports),
        ('comm_manager', test_comm_manager),
        ('config', test_config),
        ('result_dataclass', test_result_dataclass),
        ('subproblem_manager_methods', test_subproblem_manager_methods),
        ('base_estimation_manager', test_base_estimation_manager),
        ('row_generation_manager', test_row_generation_manager),
        ('oracles_manager_none_check', test_oracles_manager_none_check),
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
                print(f"  {test_name}: FAIL - {e}", flush=True)
    
    if rank == 0:
        print("\n" + "="*60, flush=True)
        print("Test Summary:", flush=True)
        print("="*60, flush=True)
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}", flush=True)
        
        all_passed = all(results.values())
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}", flush=True)
