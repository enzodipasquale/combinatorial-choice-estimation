#!/usr/bin/env python
"""Realistic usage test - follows actual user workflow"""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_realistic_workflow():
    """Test the realistic workflow: config -> data -> oracles -> subproblem -> row gen"""
    try:
        from bundlechoice.config import BundleChoiceConfig, DimensionsConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        from bundlechoice.estimation.row_generation import RowGenerationManager
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Step 1: Create config
        cfg = BundleChoiceConfig()
        cfg.dimensions.num_obs = 10
        cfg.dimensions.num_items = 5
        cfg.dimensions.num_features = 3
        cfg.dimensions.num_simulations = 1
        cfg.subproblem.name = 'Greedy'
        cfg.row_generation.max_iters = 5
        cfg.row_generation.min_iters = 1
        
        # Step 2: Initialize managers
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        # Step 3: Create dummy data
        num_agents = cfg.dimensions.num_agents
        num_items = cfg.dimensions.num_items
        num_features = cfg.dimensions.num_features
        
        if rank == 0:
            # Create dummy bundles
            obs_bundles = np.random.rand(num_agents, num_items) > 0.5
            
            # Create dummy features
            modular_agent = np.random.randn(num_agents, num_features)
            
            input_data = {
                'agent_data': {
                    'obs_bundles': obs_bundles,
                    'modular': modular_agent
                },
                'item_data': {}
            }
        else:
            input_data = None
        
        # Step 4: Load data
        dm.load_input_data(input_data if rank == 0 else {'agent_data': {}, 'item_data': {}})
        
        # Step 5: Set oracles
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        
        # Step 6: Initialize row generation
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        
        # Step 7: Try to solve (this will reveal bugs)
        try:
            result = rgm.solve(init_master=False, init_subproblems=True)
            if rank == 0:
                print("ERROR: solve() should have failed without master init", flush=True)
            return False
        except Exception as e:
            # Expected to fail without master init, but check error type
            if rank == 0:
                print(f"Expected error (no master init): {type(e).__name__}", flush=True)
        
        # Step 8: Try with proper initialization
        try:
            # This should work if all bugs are fixed
            cfg.row_generation.max_iters = 2
            result = rgm.solve(init_master=True, init_subproblems=True)
            if rank == 0:
                print(f"SUCCESS: solve() completed. Result type: {type(result)}", flush=True)
                if result is not None:
                    print(f"  converged: {result.converged}, iterations: {result.num_iterations}", flush=True)
            return True
        except Exception as e:
            if rank == 0:
                print(f"ERROR in solve(): {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            return False
            
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"ERROR in test setup: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        return False

def test_base_estimation_create_result():
    """Test _create_result method in BaseEstimationManager"""
    try:
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        from bundlechoice.estimation.base import BaseEstimationManager
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        cfg = BundleChoiceConfig()
        cfg.dimensions.num_obs = 5
        cfg.dimensions.num_items = 3
        cfg.dimensions.num_features = 2
        cfg.dimensions.num_simulations = 1
        cfg.row_generation.max_iters = 10
        
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        # Create dummy data
        if rank == 0:
            obs_bundles = np.random.rand(cfg.dimensions.num_agents, cfg.dimensions.num_items) > 0.5
            modular_agent = np.random.randn(cfg.dimensions.num_agents, cfg.dimensions.num_features)
            input_data = {
                'agent_data': {
                    'obs_bundles': obs_bundles,
                    'modular': modular_agent
                },
                'item_data': {}
            }
        else:
            input_data = None
        
        dm.load_input_data(input_data if rank == 0 else {'agent_data': {}, 'item_data': {}})
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        
        bem = BaseEstimationManager(cm, cfg, dm, om, sm)
        
        # Try to call _create_result - this will reveal the self.cfg bug
        try:
            # Create a dummy master model
            import gurobipy as gp
            if rank == 0:
                master_model = gp.Model()
                master_model.setParam('OutputFlag', 0)
                theta_sol = np.array([1.0, 2.0])
                result = bem._create_result(5, master_model, theta_sol)
                if result is None:
                    if rank == 0:
                        print("ERROR: _create_result returned None", flush=True)
                    return False
                if rank == 0:
                    print(f"SUCCESS: _create_result worked. Result type: {type(result)}", flush=True)
                return True
            else:
                result = bem._create_result(5, None, np.array([1.0, 2.0]))
                return result is None  # Non-root should return None
        except AttributeError as e:
            if 'cfg' in str(e):
                if rank == 0:
                    print(f"BUG FOUND: {e}", flush=True)
                return False
            raise
        except Exception as e:
            if rank == 0:
                print(f"ERROR in _create_result: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            return False
            
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"ERROR in test setup: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        return False

def test_row_generation_initialize_master():
    """Test _initialize_master_problem method signature"""
    try:
        from bundlechoice.config import BundleChoiceConfig
        from bundlechoice.comm_manager import CommManager
        from bundlechoice.data_manager import DataManager
        from bundlechoice.oracles_manager import OraclesManager
        from bundlechoice.subproblems.subproblem_manager import SubproblemManager
        from bundlechoice.estimation.row_generation import RowGenerationManager
        import inspect
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        cfg = BundleChoiceConfig()
        cfg.dimensions.num_obs = 5
        cfg.dimensions.num_items = 3
        cfg.dimensions.num_features = 2
        cfg.dimensions.num_simulations = 1
        
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        # Create dummy data
        if rank == 0:
            obs_bundles = np.random.rand(cfg.dimensions.num_agents, cfg.dimensions.num_items) > 0.5
            modular_agent = np.random.randn(cfg.dimensions.num_agents, cfg.dimensions.num_features)
            input_data = {
                'agent_data': {
                    'obs_bundles': obs_bundles,
                    'modular': modular_agent
                },
                'item_data': {}
            }
        else:
            input_data = None
        
        dm.load_input_data(input_data if rank == 0 else {'agent_data': {}, 'item_data': {}})
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        
        # Check method signature
        sig = inspect.signature(rgm._initialize_master_problem)
        params = list(sig.parameters.keys())
        
        if rank == 0:
            print(f"_initialize_master_problem parameters: {params}", flush=True)
        
        # Check if agent_weights is in the signature
        if 'agent_weights' in params:
            if rank == 0:
                print("INFO: agent_weights is in method signature", flush=True)
        else:
            # Check if solve() method passes agent_weights
            import ast
            import os
            file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
            with open(file_path, 'r') as f:
                source = f.read()
            if 'agent_weights' in source and '_initialize_master_problem' in source:
                # Check if it's passed as argument
                lines = source.split('\n')
                for i, line in enumerate(lines, 1):
                    if '_initialize_master_problem(' in line and 'agent_weights' in line:
                        if rank == 0:
                            print(f"BUG FOUND: Line {i} passes agent_weights but method doesn't accept it", flush=True)
                        return False
        
        return True
        
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        return False

def test_on_constraint_removed():
    """Test if _on_constraint_removed method exists"""
    try:
        from bundlechoice.estimation.row_generation import RowGenerationManager
        
        if hasattr(RowGenerationManager, '_on_constraint_removed'):
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("INFO: _on_constraint_removed method exists", flush=True)
            return True
        else:
            # Check if it's called
            import os
            file_path = os.path.join(os.path.dirname(__file__), '..', 'estimation', 'row_generation.py')
            with open(file_path, 'r') as f:
                source = f.read()
            if '_on_constraint_removed' in source:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("BUG FOUND: _on_constraint_removed is called but method doesn't exist", flush=True)
                return False
            return True
            
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        return False

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60, flush=True)
        print("Testing realistic usage workflow", flush=True)
        print("="*60, flush=True)
    
    results = {}
    
    tests = [
        ('realistic_workflow', test_realistic_workflow),
        ('base_estimation_create_result', test_base_estimation_create_result),
        ('row_generation_initialize_master', test_row_generation_initialize_master),
        ('on_constraint_removed', test_on_constraint_removed),
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
    
    if rank == 0:
        print("\n" + "="*60, flush=True)
        print("Test Summary:", flush=True)
        print("="*60, flush=True)
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}", flush=True)
        
        all_passed = all(results.values())
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}", flush=True)
